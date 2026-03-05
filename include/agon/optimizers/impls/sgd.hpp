#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <eve/module/core.hpp>

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace agon::optim {
  struct SGDParams {
    float lr = 0.01f;
    float momentum = 0.0f;

    bool nesterov = false;
    bool maximize = false;

    template<class Archive>
    void serialize(Archive& ar) {
      ar(lr, momentum, nesterov, maximize);
    }
  };

  template<typename DedupedTuple>
  struct SGDState : public OptimizerState {
    ExtractedVector<DedupedTuple> momentum{};
  };

  template<typename... Ts>
  class SGD : public Optimizer<Ts...> {
    public:
      explicit SGD(ParameterPack<Ts...> parameters, SGDParams options = {}, int num_proc = 1)
        : Optimizer<Ts...>(parameters), options_(options), num_proc_(num_proc) {
          std::apply([&](auto&... param_vecs) {
            ([&](auto& param_vec) {
              using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
              auto& mom = std::get<ExtractType_t<ParamType>>(this->state_.momentum);
              for (auto& param_ref : param_vec) {
                auto& param = param_ref.get();
                using T = typename ParamType::DataType;
                mom.insert(mom.end(), param.numel(), T(0));
              }
            }(param_vecs), ...);
          }, this->parameters_.data);
        }

      void step() override {
        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<ExtractType_t<ParamType>>(state_.momentum);

            size_t state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr size_t vec_size = eve::wide<T>::size();
              constexpr size_t unroll_factor = detail::UNROLL_FACTOR;

              std::vector<std::thread> threads;
              size_t chunk_size = (param.numel() + num_proc_ - 1) / num_proc_;

              for (size_t t = 0; t < num_proc_; ++t) {
                threads.emplace_back([&, t](){
                  size_t start = t * chunk_size;
                  size_t end = std::min(start + chunk_size, param.numel());

                  size_t i = start;
                  for (; i + vec_size * unroll_factor <= end; i += vec_size * unroll_factor) {
                    detail::unroll<unroll_factor>([&]<size_t index>() {
                      constexpr size_t offset = index * vec_size;

                      eve::wide<T> grad(&grad_full[i + offset]);
                      eve::wide<T> mom(&mom_full[state_offset + i + offset]);

                      if (options_.maximize) grad = -grad;

                      auto update = [&](){
                        if (!options_.momentum) return grad;

                        mom = eve::fma(eve::wide<T>(options_.momentum), mom, grad);
                        eve::store(mom, &mom_full[state_offset + i + offset]);

                        if (options_.nesterov) return eve::fma(eve::wide<T>(options_.momentum), mom, grad);
                        else return mom;
                      }();

                      eve::wide<T> data(&data_full[i + offset]);
                      data = eve::fma(eve::wide<T>(options_.lr), update, data);
                      eve::store(data, &data_full[i + offset]);
                    });
                  }

                  for (; i < end; ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                    T mom = options_.momentum * mom_full[state_offset + i] + grad;
                    mom_full[state_offset + i] = mom;

                    T update = options_.nesterov ? (options_.momentum * mom + grad) : mom;
                    data_full[i] += options_.lr * update;
                  }
                });
              }

              for (auto& thread : threads) {
                thread.join();
              }

              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
        this->state_.step++;
      }

      void load_from_bin(const std::string& path_str) override {
        std::filesystem::path path(path_str);
        path.replace_extension(".bin");

        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        cereal::BinaryInputArchive ar(in);
        std::string name;
        ar(name);
        if (name != optimizer_name()) throw std::runtime_error("Optimizer type mismatch: expected " + std::string(optimizer_name()));

        std::string dtype_str;
        ar(dtype_str);
        if (dtype_str != dtypes()) throw std::runtime_error("Optimizer data type mismatch: expected " + std::string(dtypes()));

        ar(options_, state_.step, state_.momentum);

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

      void save_to_bin(const std::string& path_str) const override {
        std::filesystem::path path(path_str);
        path.replace_extension(".bin");

        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        cereal::BinaryOutputArchive ar(out);
        std::string name(optimizer_name());
        std::string dtype_str(dtypes());

        ar(name, dtype_str, options_, state_.step, state_.momentum);

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

    private:
      SGDParams options_;
      SGDState<Ts...> state_;
      int num_proc_;

      static constexpr const char* optimizer_name() { return "sgd\0"; }
      static constexpr const char* dtypes() { return (std::string_view(typeid(Ts).name()) + ...); }
  };
}
