#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <eve/module/core.hpp>

#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace agon::optim {
  struct SVRGParams {
    float lr = 0.5f;
    int recompute_every = 64;

    bool maximize = false;
  };

  template<typename DedupedTuple>
  struct SVRGState : public OptimizerState {
    ExtractedVector<DedupedTuple> ref_exact{};
    ExtractedVector<DedupedTuple> ref_est{};
    ExtractedVector<DedupedTuple> ref_data{};

    bool use_ref = true;
  };

  template<typename... Ts>
  class SVRG : public Optimizer<Ts...> {
    public:
      explicit SVRG(ParameterPack<Ts...> parameters, SVRGParams options = {})
        : Optimizer<Ts...>(parameters), options_(options) {
          std::apply([&](auto&... param_vecs) {
            ([&](auto& param_vec) {
              using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
              auto& ref_exact = std::get<ExtractType_t<ParamType>>(this->state_.ref_exact);
              auto& ref_est = std::get<ExtractType_t<ParamType>>(this->state_.ref_est);
              auto& ref_data = std::get<ExtractType_t<ParamType>>(this->state_.ref_data);
              for (auto& param_ref : param_vec) {
                auto& param = param_ref.get();
                using T = typename ParamType::DataType;
                ref_exact.insert(ref_exact.end(), param.numel(), T(0));
                ref_est.insert(ref_est.end(), param.numel(), T(0));
                ref_data.insert(ref_data.end(), param.begin(), param.end());
              }
            }(param_vecs), ...);
          }, this->parameters_.data);
        }

      bool recompute() const override { return (options_.recompute_every != -1) && state_.step % options_.recompute_every == 0; }
      bool use_ref() const override { return state_.use_ref; }

      template<typename DedupedTuple>
      ExtractedVector<DedupedTuple>& ref_exact() { return std::get<ExtractType_t<DedupedTuple>>(state_.ref_exact); }
      template<typename DedupedTuple>
      ExtractedVector<DedupedTuple>& ref_est() { return std::get<ExtractType_t<DedupedTuple>>(state_.ref_est); }
      template<typename DedupedTuple>
      ExtractedVector<DedupedTuple>& ref_data() { return std::get<ExtractType_t<DedupedTuple>>(state_.ref_data); }

      void step() override {
        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& ref_exact_full = std::get<ExtractType_t<ParamType>>(state_.ref_exact);
            auto& ref_est_full = std::get<ExtractType_t<ParamType>>(state_.ref_est);

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
                  constexpr size_t off = index * vec_size;

                  eve::wide<T> grad(&grad_full[i + off]);
                  if (options_.maximize) grad = -grad;

                  auto update = [&](){
                    if ((options_.recompute_every != -1) && state_.step % options_.recompute_every == 0) return grad;

                    eve::wide<T> ref_exact(&ref_exact_full[state_offset + i + off]);
                    eve::wide<T> ref_est(&ref_est_full[state_offset + i + off]);

                    return eve::add(eve::sub(grad, ref_est), ref_exact);
                  };

                  eve::wide<T> data(&data_full[i + off]);
                  data = eve::fma(eve::wide<T>(options_.lr), update, data);
                  eve::store(data, &data_full[i + off]);
                });
              }

              for (; i < grad_full.size(); ++i) {
                T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                T update = ((options_.recompute_every != -1) && state_.step % options_.recompute_every == 0)
                  ? grad : grad - ref_est_full[state_offset + i] + ref_exact_full[state_offset + i];

                data_full[i] += options_.lr * update;
              }

              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
        this->state_.step++;
        state_.use_ref = !state_.use_ref;
      }

      void load_from_bin(const std::string& path_str) override {
        std::filesystem::path path(path_str + ".bin");
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
            auto& ref_exact = std::get<ExtractType_t<ParamType>>(state_.ref_exact);
            auto& ref_est = std::get<ExtractType_t<ParamType>>(state_.ref_est);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              if ((options_.recompute_every != -1)) {
                in.read(reinterpret_cast<char*>(ref_exact.data() + state_offset), param.numel() * sizeof(T));
                in.read(reinterpret_cast<char*>(ref_est.data() + state_offset), param.numel() * sizeof(T));
              }
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
            auto& ref_exact = std::get<ExtractType_t<ParamType>>(state_.ref_exact);
            auto& ref_est = std::get<ExtractType_t<ParamType>>(state_.ref_est);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              if ((options_.recompute_every != -1)) {
                out.write(reinterpret_cast<const char*>(ref_exact.data() + state_offset), param.numel() * sizeof(T));
                out.write(reinterpret_cast<const char*>(ref_est.data() + state_offset), param.numel() * sizeof(T));
              }
              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

    private:
      SVRGParams options_;
      SVRGState<std::tuple<Ts...>> state_;

      constexpr const char* optimizer_name() const { return "svrg\0"; }
  };
}