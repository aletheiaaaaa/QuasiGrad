#pragma once

#include <algorithm>
#include <cassert>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <eve/module/core.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "detail/dedup.hpp"
#include "detail/unpack.hpp"
#include "detail/utils.hpp"

namespace mirage {
struct Range {
  static constexpr int Start = 0;
  static constexpr int End = std::numeric_limits<int>::max();

  int start, end;

  Range(int idx) : start(idx), end(idx) {};
  Range(int idx0, int idx1) : start(idx0), end(idx1) {};
  Range() : start(Start), end(End) {};
};

namespace detail {
struct ViewParams {
  int offset;
  std::vector<int> shape;
  std::vector<int> strides;
};

inline ViewParams compute_view(
  std::span<const Range> slices, std::span<const int> src_shape, std::span<const int> src_strides
) {
  int offset = 0;

  std::vector<int> shape;
  std::vector<int> strides;

  for (const auto& [idx, slice] : std::views::enumerate(slices)) {
    assert((slice.end <= src_shape[idx] || slice.end == Range::End) && "Array index out of bounds");
    assert((slice.start <= slice.end + 1) && "Slice should begin before it ends");

    offset += slice.start * src_strides[idx];
    int dim_size = std::min(slice.end, src_shape[idx]) - slice.start;
    if (dim_size > 0) {
      shape.push_back(dim_size);
      strides.push_back(src_strides[idx]);
    }
  }

  return ViewParams{.offset = offset, .shape = std::move(shape), .strides = std::move(strides)};
}

inline std::vector<int> compute_view_idx(
  int offset, std::span<const int> shape, std::span<const int> strides
) {
  int numel = std::accumulate(shape.begin(), shape.end(), int{1}, std::multiplies<int>{});
  std::vector<int> indices(numel, 0);
  std::vector<int> coord(shape.size(), 0);

  int current_idx = offset;
  for (int i = 0; i < numel; ++i) {
    indices[i] = current_idx;

    for (int d = shape.size(); d-- > 0;) {
      coord[d]++;
      current_idx += strides[d];

      if (coord[d] < shape[d]) break;

      coord[d] = 0;
      current_idx -= shape[d] * strides[d];
    }
  }

  return indices;
}

inline std::vector<int> compute_transpose_idx(
  int idx0, int idx1, std::span<const int> shape, std::span<const int> strides
) {
  int numel = std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
  std::vector<int> indices = compute_view_idx(0, shape, strides);
  std::vector<int> new_indices(indices.size(), 0);

  std::vector<int> new_shape(shape.begin(), shape.end());
  std::vector<int> new_strides(strides.begin(), strides.end());

  std::swap(new_shape[idx0], new_shape[idx1]);
  std::swap(new_strides[idx0], new_strides[idx1]);

  std::vector<int> gather_indices = compute_view_idx(0, new_shape, new_strides);

  collect(
    std::span<const int>(indices),
    std::span<int>(new_indices),
    std::span<const int>(gather_indices),
    numel
  );

  return new_indices;
}
}  // namespace detail

template <typename T>
  requires std::is_floating_point_v<T>
class Parameter;

template <typename T>
  requires std::is_floating_point_v<T>
class View {
  public:
  using is_param_like = std::true_type;
  using DataType = T;

  View(Parameter<T>& ref, int offset, std::vector<int> shape, std::vector<int> strides)
    : ref_(ref), offset_(offset), shape_(std::move(shape)), strides_(std::move(strides)) {}

  template <typename... Args>
    requires(std::convertible_to<Args, Range> && ...)
  View<T> operator[](Args... args) {
    std::vector<Range> slices{args...};
    assert((slices.size() <= shape_.size()) && "There cannot be more slices than dimensions");
    while (slices.size() < shape_.size()) {
      slices.emplace_back();
    }

    auto params = detail::compute_view(slices, shape_, strides_);
    return View<T>(
      ref_.get(), offset_ + params.offset, std::move(params.shape), std::move(params.strides)
    );
  }

  template <typename... Args>
    requires(std::convertible_to<Args, Range> && ...)
  const View<const T> operator[](Args... args) const {
    std::vector<Range> slices{args...};
    assert((slices.size() <= shape_.size()) && "There cannot be more slices than dimensions");
    while (slices.size() < shape_.size()) {
      slices.emplace_back();
    }

    auto params = detail::compute_view(slices, shape_, strides_);
    return View<const T>(
      ref_.get(), offset_ + params.offset, std::move(params.shape), std::move(params.strides)
    );
  }

  std::vector<T> materialize() {
    std::vector<T> result(numel());
    detail::collect(
      std::span<const T>(data()), std::span<T>(result), std::span<const T>(indices()), numel()
    );

    return result;
  }

  void fill(const std::initializer_list<T>& new_data) {
    assert(new_data.size() == numel() && "Data size does not match view size");
    auto& idx = indices();

    constexpr int vec_size = eve::wide<T>::size();
    const int unroll_factor = detail::UNROLL_FACTOR;

    int i = 0;
    for (; i + vec_size * unroll_factor <= new_data.size(); i += vec_size * unroll_factor) {
      detail::unroll<unroll_factor>([&]<int index>() {
        constexpr int off = index * vec_size;

        eve::wide<T> vals(&new_data[i + off]);
        for (int k = 0; k < vec_size; ++k) {
          data()[idx[i + off + k]] = vals.get(k);
        }
      });
    }

    for (; i < new_data.size(); ++i) {
      data()[idx[i]] = new_data[i];
    }
  }

  template <typename S>
    requires detail::NestedSpan<S, T>
  View& operator=(const S& new_data) {
    auto new_shape = detail::deduce_shape(new_data);
    assert(new_shape == shape_ && "Cannot assign to view with data of different shape");
    auto& idx = indices();

    detail::fill(new_data, new_shape, [&](const auto& leaf, int offset) {
      int i = 0;

      constexpr int vec_size = eve::wide<T>::size();
      constexpr int unroll_factor = detail::UNROLL_FACTOR;

      for (; i + vec_size * unroll_factor <= leaf.size(); i += vec_size * unroll_factor) {
        detail::unroll<unroll_factor>([&]<int index>() {
          constexpr int off = index * vec_size;

          eve::wide<T> vals(&leaf[i + off]);
          for (int k = 0; k < vec_size; ++k) {
            data()[idx[offset + i + off + k]] = vals.get(k);
          }
        });
      }
      for (; i < leaf.size(); ++i) {
        data()[idx[offset + i]] = leaf[i];
      }
    });
    return *this;
  }

  std::vector<T>& grad() { return ref_.get().grad(); }
  const std::vector<T>& grad() const { return ref_.get().grad(); }

  std::vector<T>& data() { return ref_.get().data(); }
  const std::vector<T>& data() const { return ref_.get().data(); }

  const std::vector<int>& size() const { return shape_; }
  int size(int i) const { return shape_[i]; }

  const std::vector<int>& strides() const { return strides_; }
  int strides(int i) const { return strides_[i]; };

  int rank() const { return shape_.size(); }
  int numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), int{1}, std::multiplies<int>{});
  }

  private:
  std::reference_wrapper<Parameter<std::remove_const_t<T>>> ref_;
  int offset_;
  std::vector<int> shape_;
  std::vector<int> strides_;
  std::vector<int> indices_;

  std::vector<int>& indices() {
    if (indices_.empty() && !shape_.empty()) {
      indices_ = detail::compute_view_idx(offset_, shape_, strides_);
    }
    return indices_;
  }

  bool is_contiguous() const {
    int expected_stride = 1;
    for (int i = shape_.size(); i-- > 0;) {
      if (strides_[i] != expected_stride) return false;
      expected_stride *= shape_[i];
    }
    return true;
  }
};

template <typename T>
  requires std::is_floating_point_v<T>
class Parameter {
  public:
  using is_param_like = std::true_type;
  using is_quantized = std::false_type;
  using DataType = T;

  explicit Parameter(std::initializer_list<int> shape)
    : Parameter(std::vector<int>(shape.begin(), shape.end())) {}

  template <typename S>
    requires detail::NestedSpan<S, T>
  explicit Parameter(const S& span) {
    detail::unpack(span, shape_, data_, strides_);
    grad_.resize(data_.size());
  }

  Parameter<T> copy() const {
    Parameter<T> new_param(shape_);
    new_param.data_ = data_;
    new_param.grad_ = grad_;

    return new_param;
  }

  void view(std::span<const int> new_shape) {
    std::vector<int> new_shape_vec(new_shape.begin(), new_shape.end());
    int new_numel =
      std::accumulate(new_shape_vec.begin(), new_shape_vec.end(), int{1}, std::multiplies<int>{});
    assert(
      new_numel == data_.size() && "Total number of elements must remain the same when reshaping"
    );

    shape_ = std::move(new_shape_vec);
    strides_.resize(shape_.size());
    std::exclusive_scan(
      shape_.rbegin(), shape_.rend(), strides_.rbegin(), int{1}, std::multiplies<int>{}
    );
  }

  void contiguous() {
    if (is_contiguous()) return;

    std::vector<int> indices = detail::compute_view_idx(0, shape_, strides_);
    std::vector<T> new_data(data_.size());

    detail::collect(
      std::span<const T>(data_), std::span<T>(new_data), std::span<const int>(indices), numel()
    );
    data_ = std::move(new_data);

    std::exclusive_scan(
      shape_.rbegin(), shape_.rend(), strides_.rbegin(), int{1}, std::multiplies<int>{}
    );
  }

  void transpose(int idx0, int idx1) {
    assert(rank() > 1 && "Cannot transpose 1D tensors");
    assert((idx0 < rank() && idx1 < rank()) && "Transpose indices exceed rank");
    auto [min, max] = std::minmax(idx0, idx1);

    std::vector<int> indices = detail::compute_transpose_idx(idx0, idx1, shape_, strides_);
    std::vector<T> new_data(data_.size());

    detail::collect(
      std::span<const T>(data_), std::span<T>(new_data), std::span<const int>(indices), numel()
    );
    data_ = std::move(new_data);

    std::exclusive_scan(
      shape_.rbegin(), shape_.rend(), strides_.rbegin(), int{1}, std::multiplies<int>{}
    );
  }

  std::vector<T>& grad() { return grad_; }
  const std::vector<T>& grad() const { return grad_; }

  std::vector<T>& data() { return data_; }
  const std::vector<T>& data() const { return data_; }

  const std::vector<int>& size() const { return shape_; }
  int size(int i) const { return shape_[i]; }

  const std::vector<int>& strides() const { return strides_; }
  int strides(int i) const { return strides_[i]; };

  int rank() const { return shape_.size(); }
  int numel() const { return data_.size(); }

  template <typename... Args>
    requires(std::convertible_to<Args, Range> && ...)
  View<T> operator[](Args... args) {
    std::vector<Range> slices{args...};
    assert((slices.size() <= shape_.size()) && "There cannot be more slices than dimensions");
    while (slices.size() < shape_.size()) {
      slices.emplace_back();
    }

    auto params = detail::compute_view(slices, shape_, strides_);
    return View<T>(*this, params.offset, std::move(params.shape), std::move(params.strides));
  }

  template <typename... Args>
    requires(std::convertible_to<Args, Range> && ...)
  const View<const T> operator[](Args... args) const {
    std::vector<Range> slices{args...};
    assert((slices.size() <= shape_.size()) && "There cannot be more slices than dimensions");
    while (slices.size() < shape_.size()) {
      slices.emplace_back();
    }

    auto params = detail::compute_view(slices, shape_, strides_);
    return View<const T>(
      const_cast<Parameter<T>&>(*this),
      params.offset,
      std::move(params.shape),
      std::move(params.strides)
    );
  }

  template <typename S>
    requires detail::NestedSpan<S, T>
  void fill(const S& new_data) {
    auto new_shape = detail::deduce_shape(new_data);
    assert(new_shape == shape_ && "Cannot fill parameter with data of different shape");

    detail::fill(new_data, new_shape, [&](const auto& leaf, int offset) {
      std::copy(leaf.begin(), leaf.end(), data_.begin() + offset);
    });
  }

  void zero_grad() { std::fill(grad_.begin(), grad_.end(), T(0)); }

  void accumulate(const std::vector<T>& new_grad) {
    constexpr int vec_size = eve::wide<T>::size();
    constexpr int unroll_factor = detail::UNROLL_FACTOR;

    int i = 0;
    for (; i + vec_size * unroll_factor <= grad_.size(); i += vec_size * unroll_factor) {
      detail::unroll<unroll_factor>([&]<int index>() {
        constexpr int offset = index * vec_size;

        eve::wide<T> grad_vec(&grad_[i + offset]);
        eve::wide<T> new_vec(&new_grad[i + offset]);
        grad_vec = eve::add(grad_vec, new_vec);
        eve::store(grad_vec, &grad_[i + offset]);
      });
    }

    for (; i < grad_.size(); ++i) {
      grad_[i] += new_grad[i];
    }
  }

  void update(const std::vector<T>& new_val) { data_ = new_val; }

  void save_to_bin(
    const std::string& path_str, bool include_metadata = true, bool include_grad = false
  ) const {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryOutputArchive ar(out);
    if (include_metadata) {
      std::string dtype(dtype_name());
      ar(dtype, shape_);
    }
    ar(data_);
    if (include_grad) ar(grad_);
  }

  protected:
  std::vector<int> shape_;
  std::vector<int> strides_;
  std::vector<T> data_;
  std::vector<T> grad_;

  explicit Parameter(const std::vector<int>& shape) : shape_(std::move(shape)) {
    strides_.resize(shape_.size());
    std::exclusive_scan(
      shape_.rbegin(), shape_.rend(), strides_.rbegin(), int{1}, std::multiplies<int>{}
    );

    int num = std::accumulate(shape_.begin(), shape_.end(), int{1}, std::multiplies<int>{});
    data_.resize(num);
    grad_.resize(num);
  }

  bool is_contiguous() const {
    int expected_stride = 1;
    for (int i = shape_.size(); i-- > 0;) {
      if (strides_[i] != expected_stride) return false;
      expected_stride *= shape_[i];
    }
    return true;
  }

  static constexpr const char* dtype_name() {
    if constexpr (std::is_same_v<T, float>)
      return "float32\0";
    else if constexpr (std::is_same_v<T, double>)
      return "float64\0";
    else
      return "unknown\0";
  }
};

template <typename Q, typename T = float>
  requires(std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
class Quantized : public Parameter<T> {
  public:
  using is_param_like = std::true_type;
  using is_quantized = std::true_type;
  using QuantizedType = Q;

  explicit Quantized(std::initializer_list<int> shape, float scale = 1.0f, float zero_point = 0.0f)
    : Parameter<T>(std::vector<int>(shape.begin(), shape.end())),
      scale_(scale),
      zero_point_(zero_point) {}
  template <typename S>
    requires detail::NestedSpan<S, Q>
  explicit Quantized(const S& span, float scale = 1.0f, float zero_point = 0.0f)
    : Parameter<T>(detail::deduce_shape(span)), scale_(scale), zero_point_(zero_point) {
    detail::fill(span, this->shape_, [&](const auto& leaf, int offset) {
      T scale_cast = static_cast<T>(scale);
      T zero_point_cast = static_cast<T>(zero_point);

      constexpr int f_vec_size = eve::wide<T>::size();

      int i = 0;
      for (; i + f_vec_size <= leaf.size(); i += f_vec_size) {
        eve::wide<T> scale_vec(scale_cast);
        eve::wide<T> zero_point_vec(zero_point_cast);

        eve::wide<Q, eve::fixed<f_vec_size>> q_chunk(leaf.data() + i);
        auto q_float_vec = eve::convert(q_chunk, eve::as<T>{});

        auto val_vec = eve::sub(q_float_vec, zero_point_vec);
        val_vec = eve::mul(val_vec, scale_vec);

        eve::store(val_vec, &this->data_[offset + i]);
      }

      for (; i < leaf.size(); ++i) {
        this->data_[offset + i] = scale_cast * (static_cast<T>(leaf[i]) - zero_point_cast);
      }
    });
  }

  std::vector<Q> quantized() const {
    const auto& vals = this->data();
    std::vector<Q> quantized_data(vals.size());

    T inv_scale = static_cast<T>(1.0f / scale_);
    T zero_point_cast = static_cast<T>(zero_point_);

    constexpr int vec_size = eve::wide<T>::size();
    constexpr int unroll_factor = detail::UNROLL_FACTOR;

    int i = 0;
    for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
      detail::unroll<unroll_factor>([&]<int index>() {
        constexpr int offset = index * vec_size;

        eve::wide<T> val_vec(&vals[i + offset]);
        eve::wide<T> inv_scale_vec(inv_scale);
        eve::wide<T> zero_point_vec(zero_point_cast);
        auto q_vec = eve::fma(val_vec, inv_scale_vec, zero_point_vec);

        eve::store(eve::convert(q_vec, eve::as<Q>{}), &quantized_data[i + offset]);
      });
    }

    for (; i < vals.size(); ++i) {
      quantized_data[i] = static_cast<Q>(inv_scale * vals[i] + zero_point_cast);
    }

    return quantized_data;
  }

  std::vector<T> fake_quantized() const {
    const auto& vals = this->data();
    std::vector<T> fake_quantized_data(vals.size());

    T inv_scale = static_cast<T>(1.0f / scale_);
    T zero_point_cast = static_cast<T>(zero_point_);

    constexpr int vec_size = eve::wide<T>::size();
    constexpr int unroll_factor = detail::UNROLL_FACTOR;

    int i = 0;
    for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
      detail::unroll<unroll_factor>([&]<int index>() {
        constexpr int offset = index * vec_size;

        eve::wide<T> val_vec(&vals[i + offset]);
        eve::wide<T> scale_vec(scale_);
        eve::wide<T> inv_scale_vec(inv_scale);
        eve::wide<T> zero_point_vec(zero_point_cast);

        auto q_vec = eve::fma(val_vec, inv_scale_vec, zero_point_vec);
        q_vec = eve::nearest(q_vec);

        auto dq_vec = eve::sub(q_vec, zero_point_vec);
        dq_vec = eve::mul(dq_vec, scale_vec);
        eve::store(dq_vec, &fake_quantized_data[i + offset]);
      });
    }

    for (; i < vals.size(); ++i) {
      T q = std::round(inv_scale * vals[i] + zero_point_cast);
      fake_quantized_data[i] = scale_ * (q - zero_point_cast);
    }

    return fake_quantized_data;
  }

  float scale() const { return scale_; }
  float zero_point() const { return zero_point_; }

  void save_to_bin(
    const std::string& path_str,
    bool dequantize = false,
    bool include_metadata = true,
    bool include_grad = false
  ) const {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryOutputArchive ar(out);
    if (include_metadata) {
      std::string qtype(qtype_name());
      std::string dtype(this->dtype_name());
      ar(qtype, dtype, this->shape_, scale_, zero_point_);
    }
    if (dequantize)
      ar(fake_quantized());
    else
      ar(quantized());
    if (include_grad) ar(this->grad_);
  }

  private:
  float scale_ = 1.0f;
  float zero_point_ = 0.0f;

  std::string qtype_name() {
    if constexpr (std::is_same_v<Q, int8_t>)
      return "int8";
    else if constexpr (std::is_same_v<Q, int16_t>)
      return "int16";
    else
      return "unknown";
  }
};

namespace detail {
template <typename T>
using RefVec = std::vector<std::reference_wrapper<T>>;
}

template <typename DedupedTuple>
struct ParameterPack {
  detail::TransformTuple_t<detail::RefVec, DedupedTuple> data{};

  template <typename... Ts>
    requires(std::derived_from<Ts, Parameter<typename Ts::DataType>> && ...)
  ParameterPack(Ts&... params) {
    (std::get<detail::RefVec<Ts>>(data).emplace_back(params), ...);
  }

  template <typename T>
    requires std::derived_from<T, Parameter<typename T::DataType>>
  void add_parameter(T& param) {
    std::get<detail::RefVec<T>>(data).emplace_back(param);
  }
};
template <typename... Ts>
ParameterPack(Ts&...) -> ParameterPack<detail::DeduplicatedPack_t<std::decay_t<Ts>...>>;

namespace detail {
template <typename T>
concept ParamLike = T::is_param_like::value;

template <typename T>
concept NonConstPack = []<typename... Ts>(std::tuple<Ts...>*) {
  return ((!std::is_const_v<Ts> && ParamLike<Ts>) && ...);
}(static_cast<T*>(nullptr));

template <typename T, typename TypeTuple>
struct TaggedVector : private std::vector<T> {
  using std::vector<T>::vector;
  using std::vector<T>::operator=;
  using std::vector<T>::operator[];
  using std::vector<T>::data;
  using std::vector<T>::begin;
  using std::vector<T>::end;
  using std::vector<T>::insert;

  TaggedVector(std::vector<T>&& v) : std::vector<T>(std::move(v)) {}
  TaggedVector(const std::vector<T>& v) : std::vector<T>(v) {}

  template <class Archive>
  void save(Archive& ar) const {
    ar(static_cast<const std::vector<T>&>(*this));
  }

  template <class Archive>
  void load(Archive& ar) {
    ar(static_cast<std::vector<T>&>(*this));
  }
};

template <typename T>
struct ExtractType {};
template <typename T>
struct ExtractType<Parameter<T>> {
  using Type = TaggedVector<T, std::tuple<T>>;
};
template <typename Q, typename T>
struct ExtractType<Quantized<Q, T>> {
  using Type = TaggedVector<T, std::tuple<Q, T>>;
};
template <typename T>
using ExtractType_t = typename ExtractType<T>::Type;

template <typename T>
struct PrintType {
  static std::string name() { return "unknown"; }
};
template <typename T>
struct PrintType<Parameter<T>> {
  static std::string name() { return "Parameter<" + detail::TypeName<T>::name() + ">"; }
};
template <typename Q, typename T>
struct PrintType<Quantized<Q, T>> {
  static std::string name() {
    return "Quantized<" + detail::TypeName<Q>::name() + ", " + detail::TypeName<T>::name() + ">";
  }
};

template <typename DedupedTuple>
using ExtractedVector = detail::TransformTuple_t<ExtractType_t, DedupedTuple>;
}  // namespace detail
}  // namespace mirage