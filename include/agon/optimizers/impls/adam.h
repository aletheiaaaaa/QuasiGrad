#pragma once

#include "../optimizer.h"
#include "../../detail/simd/ops.h"
#include "../../detail/simd/utils.h"

#include <cmath>
#include <fstream>
#include <filesystem>
#include <stdexcept>

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
    detail::TransformTuple_t<std::vector, detail::TransformTuple_t<ExtractType_t, DedupedTuple>> momentum{};
    detail::TransformTuple_t<std::vector, detail::TransformTuple_t<ExtractType_t, DedupedTuple>> velocity{};
  };

  template<typename... Ts>
  class Adam : public Optimizer<Ts...> {
    public:
      explicit Adam(ParameterPack<Ts...> parameters, AdamParams options = {})
        : Optimizer<Ts...>(parameters), options_(options) {
          [&]<size_t... Is>(std::index_sequence<Is...>) {
            ([&]<size_t I>() {
              for (auto& param_ref : std::get<I>(this->parameters_.data)) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& mom = std::get<I>(this->state_.momentum);
                auto& vel = std::get<I>(this->state_.velocity);
                mom.insert(mom.end(), param.numel(), T(0));
                vel.insert(vel.end(), param.numel(), T(0));
              }
            }.template operator()<Is>(), ...);
          }(std::make_index_sequence<sizeof...(Ts)>{});
        }

      void step() override {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
          ([&]<size_t I>() {
            auto& param_vec = std::get<I>(this->parameters_.data);
            auto& mom_full = std::get<I>(state_.momentum);
            auto& vel_full = std::get<I>(state_.velocity);

            size_t state_offset = 0;
            for ([[maybe_unused]] auto [_, param_ref] : std::views::enumerate(param_vec)) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr size_t vec_size = simd::vec<T>::size;
              constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

              size_t i = 0;
              for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
                simd::unroll<unroll_factor>([&]<size_t index>() {
                  constexpr size_t offset = index * vec_size;

                  auto grad = simd::load<T>(&grad_full[i + offset]);
                  auto mom = simd::load<T>(&mom_full[state_offset + i + offset]);
                  auto vel = simd::load<T>(&vel_full[state_offset + i + offset]);

                  if (options_.maximize) grad = simd::neg(grad);

                  auto beta1 = simd::set1<T>(options_.beta1);
                  mom = simd::fmadd(beta1, mom, grad);
                  mom = simd::fnmadd(beta1, grad, mom);

                  auto beta2 = simd::set1<T>(options_.beta2);
                  auto grad_squared = (options_.use_adazo) ? simd::mul(mom, mom) : simd::mul(grad, grad);
                  vel = simd::fmadd(beta2, vel, grad_squared);
                  vel = simd::fnmadd(beta2, grad_squared, vel);

                  simd::store(&mom_full[state_offset + i + offset], mom);
                  simd::store(&vel_full[state_offset + i + offset], vel);

                  auto update = simd::div(mom, simd::add(simd::sqrt(vel), simd::set1<T>(options_.epsilon)));
                  auto data = simd::load<T>(&data_full[i + offset]);

                  if (options_.lambda) update = simd::fnmadd(simd::set1<T>(options_.lambda), data, update);
                  data = simd::fmadd(simd::set1<T>(options_.lr), update, data);
                  simd::store(&data_full[i + offset], data);
                });
              }

              for (; i < grad_full.size(); ++i) {
                T grad = options_.maximize ? -grad_full[i] : grad_full[i];

                T mom = options_.beta1 * mom_full[state_offset + i] + (1 - options_.beta1) * grad;
                T vel = options_.beta2 * vel_full[state_offset + i] + (1 - options_.beta2) * grad * grad;

                mom_full[state_offset + i] = mom;
                vel_full[state_offset + i] = vel;

                T update = mom / (std::sqrt(vel) + options_.epsilon);
                if (options_.lambda) update = -options_.lambda * data_full[i] + update;

                data_full[i] += options_.lr * update;
              }

              state_offset += param.numel();
            }
          }.template operator()<Is>(), ...);
        }(std::make_index_sequence<sizeof...(Ts)>{});

        state_.step++;
      }

      void load_from_bin(const std::string& path_str) override {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        in.read(reinterpret_cast<char*>(&options_), sizeof(options_));
        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        auto read_param = [&]<size_t... Is>(std::index_sequence<Is...>) {
          ([&]<size_t I>() {
            auto& param_vec = std::get<I>(this->parameters_.data);
            auto& mom = std::get<I>(state_.momentum);
            auto& vel = std::get<I>(state_.velocity);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(vel.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }.template operator()<Is>(), ...);
        };

        read_param(std::make_index_sequence<sizeof...(Ts)>{});
      }

      void save_to_bin(const std::string& path_str) const override {
        std::filesystem::path path(path_str);
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(reinterpret_cast<const char*>(&options_), sizeof(options_));
        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        auto write_param = [&]<size_t... Is>(std::index_sequence<Is...>) {
          ([&]<size_t I>() {
            auto& param_vec = std::get<I>(this->parameters_.data);
            auto& mom = std::get<I>(state_.momentum);
            auto& vel = std::get<I>(state_.velocity);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(vel.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }.template operator()<Is>(), ...);
        };

        write_param(std::make_index_sequence<sizeof...(Ts)>{});
      }

    private:
      AdamParams options_;
      AdamState<Ts...> state_;
  };
}
