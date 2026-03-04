#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <eve/module/core.hpp>

#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace agon::optim {
  struct SGDParams {
    float lr = 0.01f;
    float momentum = 0.0f;

    bool nesterov = false;
    bool maximize = false;
  };

  template<typename DedupedTuple>
  struct SGDState : public OptimizerState {
    ExtractedVector<DedupedTuple> momentum{};
  };

  template<typename... Ts>
  class SGD : public Optimizer<Ts...> {
    public:
      explicit SGD(ParameterPack<Ts...> parameters, SGDParams options = {})
        : Optimizer<Ts...>(parameters), options_(options) {
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
              constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

              size_t i = 0;
              for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
                simd::unroll<unroll_factor>([&]<size_t index>() {
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

              for (; i < grad_full.size(); ++i) {
                T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                T mom = options_.momentum * mom_full[state_offset + i] + grad;
                mom_full[state_offset + i] = mom;

                T update = options_.nesterov ? (options_.momentum * mom + grad) : mom;
                data_full[i] += options_.lr * update;
              }

              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
        this->state_.step++;
      }

      void load_from_bin(const std::string& path_str) override {
        std::filesystem::path path(path_str + ".bin");
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        std::string name;
        std::getline(in, name, '\0');
        if (name != optimizer_name()) throw std::runtime_error("Optimizer type mismatch: expected " + std::string(optimizer_name()));

        in.read(reinterpret_cast<char*>(&options_), sizeof(options_));
        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<ExtractType_t<ParamType>>(state_.momentum);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              if (options_.momentum) in.read(reinterpret_cast<char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

      void save_to_bin(const std::string& path_str) const override {
        std::filesystem::path path(path_str + ".bin");
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(optimizer_name(), std::strlen(optimizer_name()) + 1);
        out.write(reinterpret_cast<const char*>(&options_), sizeof(options_));
        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<ExtractType_t<ParamType>>(state_.momentum);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              if (options_.momentum) out.write(reinterpret_cast<const char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

    private:
      SGDParams options_;
      SGDState<Ts...> state_;

      constexpr const char* optimizer_name() const { return "sgd\0"; }
  };
}
