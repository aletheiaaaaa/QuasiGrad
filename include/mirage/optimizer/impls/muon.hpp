#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "../../detail/matrix.hpp"
#include "../../detail/thread.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct MuonOptions {
  float lr = 0.01f;
  float momentum = 0.9f;
  float epsilon = 1e-7;
  int newton_schulz_iters = 5;
  float lambda = 0.0f;

  int num_proc = 1;

  bool maximize = false;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, momentum, epsilon, newton_schulz_iters, lambda, maximize);
  }
};

template <typename TypeTuple>
struct MuonState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> momentum{};
  detail::ExtractedVector<TypeTuple> og_buf{};
  detail::ExtractedVector<TypeTuple> tp_buf{};
  detail::ExtractedVector<TypeTuple> step1_buf{};
  detail::ExtractedVector<TypeTuple> step2_buf{};
  detail::ExtractedVector<TypeTuple> step3_buf{};
};

template <typename DedupedPack>
class Muon : public Optimizer<DedupedPack> {
  public:
  explicit Muon(ParameterPack<DedupedPack> parameters, MuonOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options), pool_(options.num_proc) {
    detail::test_multidim(this->parameters_.data);
    detail::test_oom(this->parameters_.data, [&](auto& param) {
      auto s = std::min(param.size(0), param.strides(0));
      return 4 * param.numel() + 2 * s * s;
    });

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            auto& og_buf = std::get<detail::ExtractType_t<ParamType>>(state_.og_buf);
            auto& tp_buf = std::get<detail::ExtractType_t<ParamType>>(state_.tp_buf);
            auto& step1_buf = std::get<detail::ExtractType_t<ParamType>>(state_.step1_buf);
            auto& step2_buf = std::get<detail::ExtractType_t<ParamType>>(state_.step2_buf);
            auto& step3_buf = std::get<detail::ExtractType_t<ParamType>>(state_.step3_buf);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              int s = std::min(param.size(0), param.strides(0));
              mom.insert(mom.end(), param.numel(), T(0));
              og_buf.insert(og_buf.end(), param.numel(), T(0));
              tp_buf.insert(tp_buf.end(), param.numel(), T(0));
              step1_buf.insert(step1_buf.end(), s * s, T(0));
              step2_buf.insert(step2_buf.end(), s * s, T(0));
              step3_buf.insert(step3_buf.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void step() override {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            auto& og_full = std::get<detail::ExtractType_t<ParamType>>(state_.og_buf);
            auto& tp_full = std::get<detail::ExtractType_t<ParamType>>(state_.tp_buf);
            auto& step1_full = std::get<detail::ExtractType_t<ParamType>>(state_.step1_buf);
            auto& step2_full = std::get<detail::ExtractType_t<ParamType>>(state_.step2_buf);
            auto& step3_full = std::get<detail::ExtractType_t<ParamType>>(state_.step3_buf);

            int state_offset = 0;
            int sq_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              int width = param.size(0);
              int height = param.strides(0);
              int numel = param.numel();
              int smaller = std::min(width, height);
              int larger = std::max(width, height);

              auto og_mom_slice = std::span(mom_full).subspan(state_offset, numel);
              auto og_buf_slice = std::span(og_full).subspan(state_offset, numel);
              auto tp_buf_slice = std::span(tp_full).subspan(state_offset, numel);
              auto step1_slice = std::span(step1_full).subspan(sq_offset, smaller * smaller);
              auto step2_slice = std::span(step2_full).subspan(sq_offset, smaller * smaller);
              auto step3_slice = std::span(step3_full).subspan(state_offset, numel);

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr int vec_size = eve::wide<T>::size();

              const auto chunks = [&](int chunk_width, int chunk_height) {
                if (options_.num_proc % 2) {
                  return std::make_pair(
                    (chunk_width + options_.num_proc - 1) / options_.num_proc, chunk_height
                  );
                }
                return std::make_pair(
                  (2 * chunk_width + options_.num_proc - 1) / options_.num_proc,
                  (chunk_height + 1) / 2
                );
              };

              const auto offsets = [&](int id, int chunk_width, int chunk_height) {
                if (options_.num_proc % 2) {
                  return std::make_pair(id * chunk_width, 0);
                }
                return std::make_pair((id / 2) * chunk_width, (id % 2) * chunk_height);
              };

              const auto [sq_width, sq_height] = chunks(smaller, smaller);
              const auto [rc_width, rc_height] = chunks(smaller, larger);
              const auto [og_width, og_height] = chunks(width, height);

              pool_.run(
                [&](int i) {
                  const auto [x_off, y_off] = offsets(i, og_width, og_height);
                  detail::fma_tile(
                    std::span<const T>(grad_full),
                    std::span<T>(og_mom_slice),
                    width,
                    height,
                    std::min(og_width, width - x_off),
                    std::min(og_height, height - y_off),
                    x_off,
                    y_off,
                    options_.momentum
                  );
                },
                options_.num_proc
              );

              float norm = std::sqrt(
                std::transform_reduce(
                  og_mom_slice.begin(), og_mom_slice.end(), 0.0f, std::plus{}, [](float x) {
                    return x * x;
                  }
                )
              );

              pool_.run(
                [&](int i) {
                  const auto [x_off, y_off] = offsets(i, og_width, og_height);
                  detail::normalize(
                    og_mom_slice,
                    norm + options_.epsilon,
                    width,
                    height,
                    std::min(og_width, width - x_off),
                    std::min(og_height, height - y_off),
                    x_off,
                    y_off
                  );
                },
                options_.num_proc
              );

              if (width > height) {
                detail::transpose(
                  std::span<const T>(og_mom_slice), std::span<T>(og_buf_slice), height, width
                );
                std::copy(og_mom_slice.begin(), og_mom_slice.end(), tp_buf_slice.begin());
              } else {
                std::copy(og_mom_slice.begin(), og_mom_slice.end(), og_buf_slice.begin());
                detail::transpose(
                  std::span<const T>(og_mom_slice), std::span<T>(tp_buf_slice), height, width
                );
              }

              for (int iter = 0; iter < options_.newton_schulz_iters; ++iter) {
                std::fill(step1_slice.begin(), step1_slice.end(), T(0));
                pool_.run(
                  [&](int i) {
                    const auto [sq_x_off, sq_y_off] = offsets(i, sq_width, sq_height);

                    detail::symmetrized_tile(
                      std::span<const T>(og_buf_slice),
                      std::span<const T>(tp_buf_slice),
                      step1_slice,
                      smaller,
                      larger,
                      std::min(sq_width, smaller - sq_x_off),
                      std::min(sq_height, smaller - sq_y_off),
                      sq_x_off,
                      sq_y_off
                    );
                  },
                  options_.num_proc
                );

                std::fill(step2_slice.begin(), step2_slice.end(), T(0));
                pool_.run(
                  [&](int i) {
                    const auto [sq_x_off, sq_y_off] = offsets(i, sq_width, sq_height);

                    detail::quadratic_tile(
                      std::span<const T>(step1_slice),
                      step2_slice,
                      -4.7750f,
                      2.0315f,
                      smaller,
                      std::min(sq_width, smaller - sq_x_off),
                      std::min(sq_height, smaller - sq_y_off),
                      sq_x_off,
                      sq_y_off
                    );
                  },
                  options_.num_proc
                );

                std::fill(step3_slice.begin(), step3_slice.end(), T(0));
                pool_.run(
                  [&](int i) {
                    const auto [rc_x_off, rc_y_off] = offsets(i, rc_width, rc_height);

                    detail::ns_final_tile(
                      std::span<const T>(step2_slice),
                      std::span<const T>(og_buf_slice),
                      step3_slice,
                      3.4445f,
                      smaller,
                      larger,
                      std::min(rc_width, smaller - rc_x_off),
                      std::min(rc_height, larger - rc_y_off),
                      rc_x_off,
                      rc_y_off
                    );
                  },
                  options_.num_proc
                );

                std::copy(step3_slice.begin(), step3_slice.end(), og_buf_slice.begin());
                detail::transpose(
                  std::span<const T>(step3_slice), std::span<T>(tp_buf_slice), smaller, larger
                );
              }

              if (width > height) {
                detail::transpose(std::span<const T>(og_buf_slice), std::span<T>(og_mom_slice), width, height);
              } else {
                std::copy(og_buf_slice.begin(), og_buf_slice.end(), og_mom_slice.begin());
              }

              auto update_slice = (width > height) ? og_buf_slice : tp_buf_slice;

              pool_.run(
                [&](int i) {
                  const auto [x_off, y_off] = offsets(i, og_width, og_height);
                  if (options_.lambda) {
                    detail::fma_tile(
                      std::span<const T>(data_full),
                      update_slice,
                      width,
                      height,
                      og_width,
                      og_height,
                      x_off,
                      y_off,
                      -options_.lambda
                    );
                  }

                  detail::fma_tile(
                    std::span<const T>(update_slice),
                    std::span(data_full),
                    width,
                    height,
                    og_width,
                    og_height,
                    x_off,
                    y_off,
                    options_.lr
                  );
                },
                options_.num_proc
              );

              state_offset += numel;
              sq_offset += smaller * smaller;
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
    state_.step++;
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

    ar(options_, state_.step, state_.momentum);

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
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

    ar(name, options_, state_.step, state_.momentum);

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  private:
  MuonOptions options_;
  MuonState<DedupedPack> state_;
  detail::ThreadPool pool_;

  std::string optimizer_type() const override {
    return "Muon<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  // namespace mirage::optim
