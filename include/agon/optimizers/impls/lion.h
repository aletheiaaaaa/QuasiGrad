#pragma once

#include "../optimizer.h"
#include "../../detail/simd/ops.h"
#include "../../detail/simd/utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace agon::optim {
  struct LionParams {
    float lr = 1e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.4f;
    float epsilon = 1e-8;
    float lambda = 0.0f;

    bool maximize = false;
  };

  template<typename DedupedTuple>
  struct LionState : OptimizerState {
    detail::TransformTuple_t<std::vector, detail::TransformTuple_t<ExtractType_t, DedupedTuple>> momentum{};
  };

  template<typename... Ts>
  class Lion : public Optimizer<Ts...> {
    public:
      explicit Lion(ParameterPack<Ts...> parameters, LionParams options = {})
        : Optimizer<Ts...>(parameters), options_(options) {
          [&]<size_t... Is>(std::index_sequence<Is...>) {
            ([&]<size_t I>() {
              for (auto& param_ref : std::get<I>(this->parameters_.data)) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& mom = std::get<I>(this->state_.momentum);
                mom.insert(mom.end(), param.numel(), T(0));
              }
            }.template operator()<Is>(), ...);
          }(std::make_index_sequence<sizeof...(Ts)>{});
        }

      void step() override {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
          ([&]<size_t I>() {
            auto& param_vec = std::get<I>(this->parameters_.data);
            auto& mom_full = std::get<I>(state_.momentum);

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
                simd::unroll<unroll_factor>([&]<size_t index>(){
                  constexpr size_t offset = index * vec_size;

                  auto grad = simd::load<T>(&grad_full[i + offset]);
                  auto mom = simd::load<T>(&mom_full[state_offset + i + offset]);

                  if (options_.maximize) grad = simd::neg(grad);

                  auto beta1 = simd::set1<T>(options_.beta1);
                  auto update = simd::fmadd(beta1, mom, grad);
                  update = simd::fnmadd(beta1, grad, mom);

                  auto data = simd::load<T>(&data_full[i + offset]);

                  if (options_.lambda) update = simd::fnmadd(simd::set1<T>(options_.lambda), data, update);
                  data = simd::fmadd(simd::set1<T>(options_.lr), simd::sign(update), data);
                  simd::store(&data_full[i + offset], data);

                  auto beta2 = simd::set1<T>(options_.beta2);
                  mom = simd::fmadd(beta2, mom, grad);
                  mom = simd::fnmadd(beta2, grad, mom);
                  simd::store(&mom_full[state_offset + i + offset], mom);
                });
              }

              for (; i < grad_full.size(); ++i) {
                T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                T mom = options_.beta1 * mom_full[state_offset + i] + (1 - options_.beta1) * grad;

                T update = std::copysign(options_.lr, mom);
                if (options_.lambda) update = -options_.lambda * data_full[i] + update;

                data_full[i] += update;
                mom_full[state_offset + i] = options_.beta2 * mom_full[state_offset + i] + (1 - options_.beta2) * grad;
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

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(mom.data() + state_offset), param.numel() * sizeof(T));
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

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }.template operator()<Is>(), ...);
        };

        write_param(std::make_index_sequence<sizeof...(Ts)>{});
      }

    private:
      LionParams options_;
      LionState<Ts...> state_;
  };
}
