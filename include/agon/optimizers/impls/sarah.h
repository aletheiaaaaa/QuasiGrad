#pragma once

#include "../optimizer.h"
#include "../../detail/simd/ops.h"
#include "../../detail/simd/utils.h"

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace agon::optim {
  struct SarahParams {
    float lr = 0.01f;

    bool recompute = false;
    int recompute_interval = 0;

    bool maximize = false;
  };

  template<typename DedupedTuple>
  struct SarahState : public OptimizerState {
    detail::TransformTuple_t<std::vector, detail::TransformTuple_t<ExtractType_t, DedupedTuple>> prev_grad{};
    detail::TransformTuple_t<std::vector, detail::TransformTuple_t<ExtractType_t, DedupedTuple>> prev_update{};
  };

  template<typename... Ts>
  class Sarah : public Optimizer<Ts...> {
    public:
      explicit Sarah(ParameterPack<Ts...> parameters, SarahParams options = {})
        : Optimizer<Ts...>(parameters), options_(options) {
          [&]<size_t... Is>(std::index_sequence<Is...>) {
            ([&]<size_t I>() {
              for (auto& param_ref : std::get<I>(this->parameters_.data)) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& prev_grad = std::get<I>(this->state_.prev_grad);
                auto& prev_update = std::get<I>(this->state_.prev_update);
                prev_grad.insert(prev_grad.end(), param.numel(), T(0));
                prev_update.insert(prev_update.end(), param.numel(), T(0));
              }
            }.template operator()<Is>(), ...);
          }(std::make_index_sequence<sizeof...(Ts)>{});
        }

      void step() override {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
          ([&]<size_t I>() {
            auto& param_vec = std::get<I>(this->parameters_.data);
            auto& prev_grad_full = std::get<I>(state_.prev_grad);
            auto& prev_update_full = std::get<I>(state_.prev_update);

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
                  if (options_.maximize) grad = simd::neg(grad);

                  auto update = [&](){
                    if (options_.recompute && state_.step % options_.recompute_interval == 0) return grad;

                    auto prev_grad = simd::load<T>(&prev_grad_full[state_offset + i + offset]);
                    auto prev_update = simd::load<T>(&prev_update_full[state_offset + i + offset]);

                    return simd::add(simd::sub(grad, prev_grad), prev_update);
                  }();

                  auto data = simd::load<T>(&data_full[i + offset]);
                  data = simd::fmadd(simd::set1<T>(options_.lr), update, data);

                  simd::store(&data_full[i + offset], data);
                  simd::store(&prev_grad_full[state_offset + i + offset], grad);
                });
              }

              for (; i < grad_full.size(); ++i) {
                T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                T update = (options_.recompute && state_.step % options_.recompute_interval == 0)
                  ? grad : grad - prev_grad_full[state_offset + i] + prev_update_full[state_offset + i];

                data_full[i] += options_.lr * update;
                prev_grad_full[state_offset + i] = grad;
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
            auto& prev_grad = std::get<I>(state_.prev_grad);
            auto& prev_update = std::get<I>(state_.prev_update);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(prev_grad.data() + state_offset), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(prev_update.data() + state_offset), param.numel() * sizeof(T));
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
            auto& prev_grad = std::get<I>(state_.prev_grad);
            auto& prev_update = std::get<I>(state_.prev_update);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(prev_grad.data() + state_offset), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(prev_update.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }.template operator()<Is>(), ...);
        };

        write_param(std::make_index_sequence<sizeof...(Ts)>{});
      }

    private:
      SarahParams options_;
      SarahState<Ts...> state_;
  };
}
