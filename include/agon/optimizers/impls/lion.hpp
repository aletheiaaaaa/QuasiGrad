#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <eve/module/core.hpp>

#include <cmath>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace agon::optim {
  struct LionParams {
    float lr = 1e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.9f;
    float epsilon = 1e-8;
    float lambda = 0.0f;

    bool maximize = false;

    template<class Archive>
    void serialize(Archive& ar) {
      ar(lr, beta1, beta2, epsilon, lambda, maximize);
    }
  };

  template<typename DedupedTuple>
  struct LionState : OptimizerState {
    ExtractedVector<DedupedTuple> momentum{};
  };

  template<typename... Ts>
  class Lion : public Optimizer<Ts...> {
    public:
      explicit Lion(ParameterPack<Ts...> parameters, LionParams options = {}, int num_proc = 1)
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
                    detail::unroll<unroll_factor>([&]<size_t index>(){
                      constexpr size_t offset = index * vec_size;

                      eve::wide<T> grad(&grad_full[i + offset]);
                      eve::wide<T> mom(&mom_full[state_offset + i + offset]);

                      if (options_.maximize) grad = -grad;

                      eve::wide<T> beta1(options_.beta1);
                      auto update = eve::fma(beta1, mom, grad);
                      update = eve::fnma(beta1, grad, mom);

                      eve::wide<T> data(&data_full[i + offset]);

                      if (options_.lambda) update = eve::fnma(eve::wide<T>(options_.lambda), data, update);
                      data = eve::fma(eve::wide<T>(options_.lr), eve::sign(update), data);
                      eve::store(data, &data_full[i + offset]);

                      eve::wide<T> beta2(options_.beta2);
                      mom = eve::fma(beta2, mom, grad);
                      mom = eve::fnma(beta2, grad, mom);
                      eve::store(mom, &mom_full[state_offset + i + offset]);
                    });
                  }

                  for (; i < end; ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                    T mom = options_.beta1 * mom_full[state_offset + i] + (1 - options_.beta1) * grad;

                    T update = std::copysign(options_.lr, mom);
                    if (options_.lambda) update = -options_.lambda * data_full[i] + update;

                    data_full[i] += update;
                    mom_full[state_offset + i] = options_.beta2 * mom_full[state_offset + i] + (1 - options_.beta2) * grad;
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
      LionParams options_;
      LionState<Ts...> state_;
      int num_proc_;

      static constexpr const char* optimizer_name() { return "lion\0"; }
      static constexpr const char* dtypes() { return (std::string_view(typeid(Ts).name()) + ...); }
  };
}
