#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace agon::optim {
  struct SVRGOptions {
    float lr = 0.5f;
    float lambda = 0.0f;
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

  template<typename DedupedTuple>
  class SVRG : public Optimizer<DedupedTuple> {
    public:
      explicit SVRG(ParameterPack<DedupedTuple> parameters, SVRGOptions options = {}, int num_proc = 1)
        : Optimizer<DedupedTuple>(parameters), options_(options), num_proc_(num_proc) {
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

      template<typename T>
      ExtractedVector<T>& ref_exact() { return std::get<ExtractType_t<T>>(state_.ref_exact); }
      template<typename T>
      ExtractedVector<T>& ref_est() { return std::get<ExtractType_t<T>>(state_.ref_est); }
      template<typename T>
      ExtractedVector<T>& ref_data() { return std::get<ExtractType_t<T>>(state_.ref_data); }

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
                      constexpr size_t off = index * vec_size;

                      eve::wide<T> grad(&grad_full[i + off]);
                      eve::wide<T> data(&data_full[i + off]);
                      if (options_.maximize) grad = -grad;

                      auto update = [&](){
                        if ((options_.recompute_every != -1) && state_.step % options_.recompute_every == 0) return grad;

                        eve::wide<T> ref_exact(&ref_exact_full[state_offset + i + off]);
                        eve::wide<T> ref_est(&ref_est_full[state_offset + i + off]);

                        grad = eve::add(eve::sub(grad, ref_est), ref_exact);
                        if (options_.lambda) grad = eve::fnma(eve::wide<T>(options_.lambda), data, grad);

                        return grad;
                      }();

                      data = eve::fma(eve::wide<T>(options_.lr), update, data);
                      eve::store(data, &data_full[i + off]);
                    });
                  }

                  for (; i < end; ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                    T update = ((options_.recompute_every != -1) && state_.step % options_.recompute_every == 0)
                      ? grad : grad - ref_est_full[state_offset + i] + ref_exact_full[state_offset + i];

                    if (options_.lambda) update = update - options_.lambda * data_full[i];
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
        state_.use_ref = !state_.use_ref;
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
        if (name != optimizer_type()) throw std::runtime_error("Optimizer type mismatch: expected " + std::string(optimizer_type()));

        ar(options_, state_.step, state_.ref_exact, state_.ref_est);

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
        std::string name(optimizer_type());
        ar(name, options_, state_.step, state_.ref_exact, state_.ref_est);

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

    private:
      SVRGOptions options_;
      SVRGState<DedupedTuple> state_;
      int num_proc_;

      std::string optimizer_type() const {
        return "SVRG<" + []<typename... Us>(std::tuple<Us...>*) {
          std::string result;
          bool last = true;
          ((result += (last ? "" : ", ") + PrintType<std::remove_cvref_t<Us>>::name(), last = false), ...);
          return result;
        }(static_cast<DedupedTuple*>(nullptr)) + ">";
      }
    };
}