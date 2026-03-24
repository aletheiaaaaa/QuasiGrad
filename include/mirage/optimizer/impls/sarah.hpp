#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct SarahOptions {
  float lr = 0.01f;
  float lambda = 0.0f;
  int recompute_every = 64;

  bool maximize = false;

  int num_proc = 1;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, lambda, recompute_every, maximize);
  }
};

template <typename TypeTuple>
struct SarahState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> prev_grad{};
  detail::ExtractedVector<TypeTuple> prev_update{};
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Sarah : public Optimizer<DedupedPack> {
  public:
  explicit Sarah(ParameterPack<DedupedPack> parameters, SarahOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options) {
    if ((options_.recompute_every != -1) && options_.recompute_every == 0)
      throw std::invalid_argument(
        "Recompute every must be greater than 0 when recompute is enabled"
      );

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& prev_grad = std::get<detail::ExtractType_t<ParamType>>(this->state_.prev_grad);
            auto& prev_update =
              std::get<detail::ExtractType_t<ParamType>>(this->state_.prev_update);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              prev_grad.insert(prev_grad.end(), param.numel(), T(0));
              prev_update.insert(prev_update.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  bool recompute() const override {
    return (options_.recompute_every != -1) && state_.step % options_.recompute_every == 0;
  }

  void step() override {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& prev_grad_full = std::get<detail::ExtractType_t<ParamType>>(state_.prev_grad);
            auto& prev_update_full = std::get<detail::ExtractType_t<ParamType>>(state_.prev_update);

            int state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr int vec_size = eve::wide<T>::size();
              constexpr int unroll_factor = detail::UNROLL_FACTOR;

              std::vector<std::thread> threads;
              int chunk_size = (param.numel() + options_.num_proc - 1) / options_.num_proc;

              for (int t = 0; t < options_.num_proc; ++t) {
                threads.emplace_back([&, t]() {
                  int start = t * chunk_size;
                  int end = std::min(start + chunk_size, param.numel());

                  int i = start;
                  for (; i + vec_size * unroll_factor <= end; i += vec_size * unroll_factor) {
                    detail::unroll<unroll_factor>([&]<int index>() {
                      constexpr int offset = index * vec_size;

                      eve::wide<T> grad(&grad_full[i + offset]);
                      eve::wide<T> data(&data_full[i + offset]);
                      if (options_.maximize) grad = -grad;

                      auto update = [&]() {
                        if (
                          (options_.recompute_every != -1) &&
                          state_.step % options_.recompute_every == 0
                        )
                          return grad;

                        eve::wide<T> prev_grad(&prev_grad_full[state_offset + i + offset]);
                        eve::wide<T> prev_update(&prev_update_full[state_offset + i + offset]);

                        grad = eve::add(eve::sub(grad, prev_grad), prev_update);
                        if (options_.lambda)
                          grad = eve::fnma(eve::wide<T>(options_.lambda), data, grad);

                        return grad;
                      }();

                      data = eve::fma(eve::wide<T>(options_.lr), update, data);

                      eve::store(data, &data_full[i + offset]);
                      eve::store(grad, &prev_grad_full[state_offset + i + offset]);
                    });
                  }

                  for (; i < end; ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                    T update = ((options_.recompute_every != -1) &&
                                state_.step % options_.recompute_every == 0)
                                 ? grad
                                 : grad - prev_grad_full[state_offset + i] +
                                     prev_update_full[state_offset + i];

                    if (options_.lambda) update = update - options_.lambda * data_full[i];

                    data_full[i] += options_.lr * update;
                    prev_grad_full[state_offset + i] = grad;
                    prev_update_full[state_offset + i] = update;
                  }
                });
              }

              for (auto& thread : threads) {
                thread.join();
              }

              state_offset += param.numel();
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
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
    if (name != optimizer_type()) this->handle_type_error(name);

    ar(options_, state_.step, state_.prev_grad, state_.prev_update);

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
              ar(param_ref.get().grad());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void save_to_bin(const std::string& path_str) const override {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryOutputArchive ar(out);
    std::string name(optimizer_type());
    ar(name, options_, state_.step, state_.prev_grad, state_.prev_update);

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
              ar(param_ref.get().grad());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  std::string optimizer_type() const override {
    std::string type = "SARAH<";
    bool first = true;

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;

            if (!first) type += ", ";
            first = false;
            type += detail::PrintType<ParamType>::name() + "[";

            bool pfirst = true;
            for (auto& param_ref : param_vec) {
              if (!pfirst) type += ",";
              pfirst = false;

              auto& shape = param_ref.get().size();
              for (int i = 0; i < shape.size(); ++i) {
                if (i > 0) type += "x";
                type += std::to_string(shape[i]);
              }
            }

            type += "]";
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );

    type += ">";
    return type;
  }

  private:
  SarahOptions options_;
  SarahState<DedupedPack> state_;
};
}  // namespace mirage::optim
