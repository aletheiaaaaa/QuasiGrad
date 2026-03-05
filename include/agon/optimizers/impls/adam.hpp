#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <eve/module/core.hpp>

#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace agon::optim {
  struct AdamParams {
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.4f;
    float epsilon = 1e-8f;
    float lambda = 0.0f;

    bool maximize = false;
    bool use_adazo = false;
  };

  template<typename DedupedTuple>
  struct AdamState : public OptimizerState {
    ExtractedVector<DedupedTuple> momentum{};
    ExtractedVector<DedupedTuple> velocity{};
  };

  template<typename... Ts>
  class Adam : public Optimizer<Ts...> {
    public:
      explicit Adam(ParameterPack<Ts...> parameters, AdamParams options = {}, int num_proc = 1)
        : Optimizer<Ts...>(parameters), options_(options), num_proc_(num_proc) {
          std::apply([&](auto&... param_vecs) {
            ([&](auto& param_vec) {
              using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
              auto& mom = std::get<ExtractType_t<ParamType>>(this->state_.momentum);
              auto& vel = std::get<ExtractType_t<ParamType>>(this->state_.velocity);
              for (auto& param_ref : param_vec) {
                auto& param = param_ref.get();
                using T = typename ParamType::DataType;
                mom.insert(mom.end(), param.numel(), T(0));
                vel.insert(vel.end(), param.numel(), T(0));
              }
            }(param_vecs), ...);
          }, this->parameters_.data);
        }

      void step() override {
        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<ExtractType_t<ParamType>>(state_.momentum);
            auto& vel_full = std::get<ExtractType_t<ParamType>>(state_.velocity);

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
                threads.emplace_back([&]() {
                  size_t start = t * chunk_size;
                  size_t end = std::min(start + chunk_size, param.numel());

                  size_t i = start;
                  for (; i + vec_size * unroll_factor <= end; i += vec_size * unroll_factor) {
                    detail::unroll<unroll_factor>([&]<size_t index>() {
                      constexpr size_t offset = index * vec_size;

                      eve::wide<T> grad(&grad_full[i + offset]);
                      eve::wide<T> mom(&mom_full[state_offset + i + offset]);
                      eve::wide<T> vel(&vel_full[state_offset + i + offset]);

                      if (options_.maximize) grad = -grad;

                      eve::wide<T> beta1(options_.beta1);
                      mom = eve::fma(beta1, mom, grad);
                      mom = eve::fnma(beta1, grad, mom);

                      eve::wide<T> beta2(options_.beta2);
                      auto grad_squared = (options_.use_adazo) ? eve::mul(mom, mom) : eve::mul(grad, grad);
                      vel = eve::fma(beta2, vel, grad_squared);
                      vel = eve::fnma(beta2, grad_squared, vel);

                      eve::store(mom, &mom_full[state_offset + i + offset]);
                      eve::store(vel, &vel_full[state_offset + i + offset]);

                      auto update = eve::div(mom, eve::add(eve::sqrt(vel), eve::wide<T>(options_.epsilon)));
                      eve::wide<T> data(&data_full[i + offset]);

                      if (options_.lambda) update = eve::fnma(eve::wide<T>(options_.lambda), data, update);
                      data = eve::fma(eve::wide<T>(options_.lr), update, data);
                      eve::store(data, &data_full[i + offset]);
                    });
                  }

                  for (; i < end; ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];

                    T mom = options_.beta1 * mom_full[state_offset + i] + (1 - options_.beta1) * grad;
                    T vel = options_.beta2 * vel_full[state_offset + i] + (1 - options_.beta2) * grad * grad;

                    mom_full[state_offset + i] = mom;
                    vel_full[state_offset + i] = vel;

                    T update = mom / (std::sqrt(vel) + options_.epsilon);
                    if (options_.lambda) update = -options_.lambda * data_full[i] + update;

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
            auto& vel = std::get<ExtractType_t<ParamType>>(state_.velocity);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(vel.data() + state_offset), param.numel() * sizeof(T));
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
            auto& vel = std::get<ExtractType_t<ParamType>>(state_.velocity);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(vel.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

    private:
      AdamParams options_;
      AdamState<Ts...> state_;
      int num_proc_;

      constexpr const char* optimizer_name() const { return "adam\0"; }
  };
}
