#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../detail/matrix.hpp"
#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct SoapOptions {
  float lr = 1e-3f;
  float beta1 = 0.95f;
  float beta2 = 0.4f;
  float epsilon = 1e-8f;
  float lambda = 0.0f;
  int decompose_every = 64;

  int num_proc = 1;

  bool maximize = false;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, beta1, beta2, epsilon, lambda, decompose_every, maximize);
  }
};

template <typename TypeTuple>
struct SoapState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> momentum{};
  detail::ExtractedVector<TypeTuple> velocity{};
  detail::ExtractedVector<TypeTuple> left_velocity{};
  detail::ExtractedVector<TypeTuple> right_velocity{};
  detail::ExtractedVector<TypeTuple> left_eigenvectors{};
  detail::ExtractedVector<TypeTuple> right_eigenvectors{};
};

template <typename DedupedPack>
class Soap : public Optimizer<DedupedPack> {
  // TODO
};
}  // namespace mirage::optim