#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "../../detail/matrix.hpp"
#include "../../detail/thread.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct SPlusOptions {
  float lr = 0.5f;
  float beta1 = 0.9f;
  float beta2 = 0.4f;
  float beta3 = 0.999f;
  float lambda = 0.0f;
  int decompose_every = 64;

  int num_proc = 1;

  bool maximize = false;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, beta1, beta2, beta3, lambda, decompose_every, maximize);
  }
};

template <typename TypeTuple>
struct SPlusState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> momentum{};
  detail::ExtractedVector<TypeTuple> left_velocity{};
  detail::ExtractedVector<TypeTuple> right_velocity{};
  detail::ExtractedVector<TypeTuple> left_eigenvectors{};
  detail::ExtractedVector<TypeTuple> right_eigenvectors{};
  detail::ExtractedVector<TypeTuple> param_ema{};

  detail::ExtractedVector<TypeTuple> tpg{};
  detail::ExtractedVector<TypeTuple> leig_tp{};
  detail::ExtractedVector<TypeTuple> reig_tp{};
  detail::ExtractedVector<TypeTuple> rotated{};
  detail::ExtractedVector<TypeTuple> update{};
  detail::ExtractedVector<TypeTuple> scratch{};  

};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class SPlus : public Optimizer<DedupedPack> {
  public:
  explicit SPlus(ParameterPack<DedupedPack> parameters, SPlusOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options), pool_(options.num_proc) {
    detail::test_multidim(this->parameters_.data);
    detail::test_oom(this->parameters_.data, [&](auto& param) {

      return 5 * param.numel() +
             3 * (param.size(0) * param.size(0) + param.strides(0) * param.strides(0));
    });

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            auto& lvel = std::get<detail::ExtractType_t<ParamType>>(state_.left_velocity);
            auto& rvel = std::get<detail::ExtractType_t<ParamType>>(state_.right_velocity);
            auto& leig = std::get<detail::ExtractType_t<ParamType>>(state_.left_eigenvectors);
            auto& reig = std::get<detail::ExtractType_t<ParamType>>(state_.right_eigenvectors);
            auto& ema = std::get<detail::ExtractType_t<ParamType>>(state_.param_ema);
            auto& tpg = std::get<detail::ExtractType_t<ParamType>>(state_.tpg);
            auto& leig_tp = std::get<detail::ExtractType_t<ParamType>>(state_.leig_tp);
            auto& reig_tp = std::get<detail::ExtractType_t<ParamType>>(state_.reig_tp);
            auto& rotated = std::get<detail::ExtractType_t<ParamType>>(state_.rotated);
            auto& update = std::get<detail::ExtractType_t<ParamType>>(state_.update);
            auto& scratch = std::get<detail::ExtractType_t<ParamType>>(state_.scratch);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              int l_numel = param.size(0) * param.size(0);
              int r_numel = param.strides(0) * param.strides(0);

              using T = typename ParamType::DataType;
              mom.insert(mom.end(), param.numel(), T(0));
              lvel.insert(lvel.end(), l_numel, T(0));
              rvel.insert(rvel.end(), r_numel, T(0));
              leig.insert(leig.end(), l_numel, T(0));
              reig.insert(reig.end(), r_numel, T(0));
              ema.insert(ema.end(), param.numel(), T(0));
              tpg.insert(tpg.end(), param.numel(), T(0));
              leig_tp.insert(leig_tp.end(), l_numel, T(0));
              reig_tp.insert(reig_tp.end(), r_numel, T(0));
              rotated.insert(rotated.end(), param.numel(), T(0));
              update.insert(update.end(), param.numel(), T(0));
              scratch.insert(scratch.end(), param.numel(), T(0));
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
            auto& lvel_full = std::get<detail::ExtractType_t<ParamType>>(state_.left_velocity);
            auto& rvel_full = std::get<detail::ExtractType_t<ParamType>>(state_.right_velocity);
            auto& leig_full = std::get<detail::ExtractType_t<ParamType>>(state_.left_eigenvectors);
            auto& reig_full = std::get<detail::ExtractType_t<ParamType>>(state_.right_eigenvectors);
            auto& ema_full = std::get<detail::ExtractType_t<ParamType>>(state_.param_ema);
            auto& tpg_full = std::get<detail::ExtractType_t<ParamType>>(state_.tpg);
            auto& leig_tp_full = std::get<detail::ExtractType_t<ParamType>>(state_.leig_tp);
            auto& reig_tp_full = std::get<detail::ExtractType_t<ParamType>>(state_.reig_tp);
            auto& rotated_full = std::get<detail::ExtractType_t<ParamType>>(state_.rotated);
            auto& update_full = std::get<detail::ExtractType_t<ParamType>>(state_.update);
            auto& scratch_full = std::get<detail::ExtractType_t<ParamType>>(state_.scratch);

            int state_offset = 0;
            int left_offset = 0;
            int right_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param_og = param_ref.get();
              using T = typename ParamType::DataType;
              int width = param_og.size(0);
              int height = param_og.strides(0);

              int numel_off = param_og.numel();
              int left_off = width * width;
              int right_off = height * height;

              auto mom_slice = std::span(mom_full).subspan(state_offset, numel_off);
              auto rvel_slice = std::span(rvel_full).subspan(right_offset, right_off);
              auto lvel_slice = std::span(lvel_full).subspan(left_offset, left_off);
              auto reig_slice = std::span(reig_full).subspan(right_offset, right_off);
              auto leig_slice = std::span(leig_full).subspan(left_offset, left_off);
              auto ema_slice = std::span(ema_full).subspan(state_offset, numel_off);
              auto tpg_slice = std::span(tpg_full).subspan(state_offset, numel_off);
              auto leig_tp_slice = std::span(leig_tp_full).subspan(left_offset, left_off);
              auto reig_tp_slice = std::span(reig_tp_full).subspan(right_offset, right_off);
              auto rotated_slice = std::span(rotated_full).subspan(state_offset, numel_off);
              auto update_slice = std::span(update_full).subspan(state_offset, numel_off);
              auto scratch_slice = std::span(scratch_full).subspan(state_offset, numel_off);

              auto& og_grad = param_og.grad();
              auto& data = param_og.data();

              detail::transpose(
                std::span<const T>(og_grad), std::span<T>(tpg_slice), height, width
              );

              constexpr int vec_size = eve::wide<T>::size();

              const auto chunks = [&](int chunk_width, int chunk_height) {
                return std::make_pair(
                  (chunk_width + options_.num_proc - 1) / options_.num_proc, chunk_height
                );
              };

              const auto offsets = [&](int i, int chunk_width, int chunk_height) {
                return std::make_pair(i * chunk_width, 0);
              };

              const auto [wh_width, wh_height] = chunks(width, height);
              const auto [ww_width, ww_height] = chunks(width, width);
              const auto [hh_width, hh_height] = chunks(height, height);

              detail::transpose(
                std::span<const T>(leig_slice), std::span<T>(leig_tp_slice), width, width
              );
              detail::transpose(
                std::span<const T>(reig_slice), std::span<T>(reig_tp_slice), height, height
              );

              pool_.run(
                [&](int i) {
                  const auto [wh_x_off, wh_y_off] = offsets(i, wh_width, wh_height);
                  const auto [ww_x_off, ww_y_off] = offsets(i, ww_width, ww_height);
                  const auto [hh_x_off, hh_y_off] = offsets(i, hh_width, hh_height);

                  detail::symmetrized_ema_tile(
                    std::span<const T>(og_grad),
                    std::span<const T>(tpg_slice),
                    lvel_slice,
                    width,
                    height,
                    std::min(ww_width, width - ww_x_off),
                    std::min(ww_height, width - ww_y_off),
                    ww_x_off,
                    ww_y_off,
                    options_.beta2
                  );

                  detail::symmetrized_ema_tile(
                    std::span<const T>(tpg_slice),
                    std::span<const T>(og_grad),
                    rvel_slice,
                    height,
                    width,
                    std::min(hh_width, height - hh_x_off),
                    std::min(hh_height, height - hh_y_off),
                    hh_x_off,
                    hh_y_off,
                    options_.beta2
                  );

                  detail::pair_tile(
                    std::span<const T>(leig_tp_slice),
                    std::span<const T>(og_grad),
                    scratch_slice,
                    width,
                    width,
                    height,
                    wh_width,
                    wh_height,
                    wh_x_off,
                    wh_y_off
                  );

                  detail::sign_after_pair_tile(
                    std::span<const T>(scratch_slice),
                    std::span<const T>(reig_slice),
                    rotated_slice,
                    width,
                    height,
                    height,
                    wh_width,
                    wh_height,
                    wh_x_off,
                    wh_y_off,
                    options_.maximize
                  );

                  detail::ema_tile(
                    std::span<const T>(rotated_slice),
                    mom_slice,
                    width,
                    height,
                    std::min(wh_width, width - wh_x_off),
                    std::min(wh_height, height - wh_y_off),
                    wh_x_off,
                    wh_y_off,
                    options_.beta1
                  );
                },
                options_.num_proc
              );

              using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

              if (state_.step % options_.decompose_every == 0) {
                Eigen::Map<Matrix> lvel_format(lvel_slice.data(), width, width);
                Eigen::Map<Matrix> rvel_format(rvel_slice.data(), height, height);

                Eigen::SelfAdjointEigenSolver<Matrix> lvel_solver(lvel_format);
                Eigen::SelfAdjointEigenSolver<Matrix> rvel_solver(rvel_format);

                Eigen::Map<Matrix>(leig_slice.data(), width, width) = lvel_solver.eigenvectors();
                Eigen::Map<Matrix>(reig_slice.data(), height, height) = rvel_solver.eigenvectors();
              }

              pool_.run(
                [&](int i) {
                  const auto [x_off, y_off] = offsets(i, wh_width, wh_height);

                  detail::sign_before_pair_tile(
                    std::span<const T>(leig_slice),
                    std::span<const T>(mom_slice),
                    scratch_slice,
                    width,
                    width,
                    height,
                    wh_width,
                    wh_height,
                    x_off,
                    y_off
                  );

                  detail::norm_pair_tile(
                    std::span<const T>(scratch_slice),
                    std::span<const T>(reig_tp_slice),
                    update_slice,
                    width,
                    height,
                    height,
                    wh_width,
                    wh_height,
                    x_off,
                    y_off
                  );

                  detail::fma_tile(
                    std::span<const T>(update_slice),
                    ema_slice,
                    width,
                    height,
                    wh_width,
                    wh_height,
                    x_off,
                    y_off,
                    options_.lr
                  );

                  if (options_.lambda) {
                    detail::fma_tile(
                      std::span<const T>(data),
                      ema_slice,
                      width,
                      height,
                      wh_width,
                      wh_height,
                      x_off,
                      y_off,
                      -options_.lambda
                    );
                  }

                  detail::ema_tile(
                    std::span<const T>(ema_slice),
                    std::span(data),
                    width,
                    height,
                    wh_width,
                    wh_height,
                    x_off,
                    y_off,
                    options_.beta3
                  );
                },
                options_.num_proc
              );

              state_offset += numel_off;
              left_offset += left_off;
              right_offset += right_off;
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

    ar(
      options_,
      state_.step,
      state_.momentum,
      state_.left_velocity,
      state_.right_velocity,
      state_.left_eigenvectors,
      state_.right_eigenvectors,
      state_.param_ema
    );

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

    ar(
      name,
      options_,
      state_.step,
      state_.momentum,
      state_.left_velocity,
      state_.right_velocity,
      state_.left_eigenvectors,
      state_.right_eigenvectors,
      state_.param_ema
    );

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
  SPlusOptions options_;
  SPlusState<DedupedPack> state_;
  detail::ThreadPool pool_;

  std::string optimizer_type() const override {
    return "SPlus<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  

