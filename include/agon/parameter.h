#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <typeinfo>
#include <span>
#include <limits>
#include <ranges>
#include <cassert>
#include <numeric>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <fstream>

#include "detail/simd/utils.h"
#include "detail/simd/ops.h"
#include "detail/dedup.h"
#include "detail/unpack.h"

namespace agon {
  struct Slice {
    static constexpr size_t Start = 0;
    static constexpr size_t End = std::numeric_limits<size_t>::max();

    size_t start, end;

    Slice(size_t idx) : start(idx), end(idx) {};
    Slice(size_t idx0, size_t idx1) : start(idx0), end(idx1) {};
    Slice() : start(Start), end(End) {};
  };

  namespace detail {
    struct ViewParams {
      size_t offset;
      std::vector<size_t> shape;
      std::vector<size_t> strides;
    };

    inline ViewParams compute_view(
      std::span<const Slice> slices,
      std::span<const size_t> src_shape,
      std::span<const size_t> src_strides
    ) {
      size_t offset = 0;

      std::vector<size_t> shape;
      std::vector<size_t> strides;

      for (const auto& [idx, slice] : std::views::enumerate(slices)) {
        assert((slice.end <= src_shape[idx] || slice.end == Slice::End) && "Array index out of bounds");
        assert((slice.start <= slice.end + 1) && "Slice should begin before it ends");

        offset += slice.start * src_strides[idx];
        size_t dim_size = std::min(slice.end, src_shape[idx]) - slice.start;
        if (dim_size > 0) {
          shape.push_back(dim_size);
          strides.push_back(src_strides[idx]);
        }
      }

      return ViewParams{
        .offset = offset,
        .shape = std::move(shape),
        .strides = std::move(strides)
      };
    }

    inline std::vector<int> compute_indices(
      size_t offset,
      std::span<const size_t> shape,
      std::span<const size_t> strides
    ) {
      size_t out_numel = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
      std::vector<int> indices(out_numel, 0);
      std::vector<int> coord(shape.size(), 0);

      size_t current_idx = offset;
      for (size_t i = 0; i < out_numel; ++i) {
        indices[i] = current_idx;

        for (size_t d = shape.size(); d-- > 0; ) {
          coord[d]++;
          current_idx += strides[d];

          if (coord[d] < shape[d]) break;

          coord[d] = 0;
          current_idx -= shape[d] * strides[d];
        }
      }

      return indices;
    }
  }

  template<typename T>
    requires std::is_floating_point_v<T>
  class Parameter;

  template<typename T>
    requires std::is_floating_point_v<T>
  struct View {
    using is_param_like = std::true_type;
    using DataType = T;

    std::reference_wrapper<Parameter<std::remove_const_t<T>>> ref;
    size_t offset;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<int> indices;

    std::vector<int>& ensure_indices() {
      if (indices.empty() && !shape.empty()) {
        indices = detail::compute_indices(offset, shape, strides);
      }
      return indices;
    }

    template<typename... Args>
      requires (std::convertible_to<Args, Slice> && ...)
    View<T> operator[](Args... args) {
      std::vector<Slice> slices{args...};
      assert((slices.size() <= shape.size()) && "There cannot be more slices than dimensions");
      while (slices.size() < shape.size()) {
        slices.emplace_back();
      }

      auto params = detail::compute_view(slices, shape, strides);
      return View<T>{
        .ref = ref,
        .offset = offset + params.offset,
        .shape = std::move(params.shape),
        .strides = std::move(params.strides),
      };
    }

    template<typename... Args>
      requires (std::convertible_to<Args, Slice> && ...)
    const View<const T> operator[](Args... args) const {
      std::vector<Slice> slices{args...};
      assert((slices.size() <= shape.size()) && "There cannot be more slices than dimensions");
      while (slices.size() < shape.size()) {
        slices.emplace_back();
      }

      auto params = detail::compute_view(slices, shape, strides);
      return View<const T>{
        .ref = std::cref(ref.get()),
        .offset = offset + params.offset,
        .shape = std::move(params.shape),
        .strides = std::move(params.strides),
      };
    }

    std::vector<T> materialize() {
      auto& idx = ensure_indices();
      std::vector<T> result(numel());

      constexpr size_t vec_size = simd::vec<T>::size;
      const size_t unroll_factor = simd::UNROLL_FACTOR;

      int i = 0;
      for (; i + vec_size * unroll_factor <= result.size(); i += vec_size * unroll_factor) {
        simd::unroll<unroll_factor>([&]<size_t index>() {
          constexpr size_t off = index * vec_size;

          auto vals = simd::gather<T>(&data()[0], &idx[i + off]);
          simd::store(&result[i + off], vals);
        });
      }

      for (; i < numel(); ++i) {
        result[i] = data()[idx[i]];
      }

      return result;
    }

    void fill(const std::span<T>& new_data) {
      assert(new_data.size() == numel() && "Data size does not match view size");
      auto& idx = ensure_indices();

      constexpr size_t vec_size = simd::vec<T>::size;
      const size_t unroll_factor = simd::UNROLL_FACTOR;

      int i = 0;
      for (; i + vec_size * unroll_factor <= new_data.size(); i += vec_size * unroll_factor) {
        simd::unroll<unroll_factor>([&]<size_t index>() {
          constexpr size_t off = index * vec_size;

          auto vals = simd::load<T>(&new_data[i + off]);
          simd::scatter(&data()[0], &idx[i + off], vals);
      });
      }

      for (; i < new_data.size(); ++i) {
        data()[idx[i]] = new_data[i];
      }
    }

    template<typename S>
      requires detail::NestedSpan<S, T>
    View& operator=(const S& new_data) {
      auto new_shape = detail::deduce_shape(new_data);
      assert(new_shape == shape && "Cannot assign to view with data of different shape");
      auto& idx = ensure_indices();

      detail::fill(new_data, new_shape, [&](const auto& leaf, size_t offset) {
        int i = 0;

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        for (; i + vec_size * unroll_factor <= leaf.size(); i += vec_size * unroll_factor) {
          simd::unroll<unroll_factor>([&]<size_t index>() {
            constexpr size_t off = index * vec_size;

            auto vals = simd::load<T>(&leaf[i + off]);
            simd::scatter(&data()[0], &idx[offset + i + off], vals);
          });
        }
        for (; i < leaf.size(); ++i) {
          data()[idx[offset + i]] = leaf[i];
        }
      });
      return *this;
    }

    std::vector<T>& grad() { return ref.get().grad(); }
    const std::vector<T>& grad() const { return ref.get().grad(); }

    std::vector<T>& data() { return ref.get().data(); }
    const std::vector<T>& data() const { return ref.get().data(); }

    size_t rank() const { return shape.size(); }

    size_t numel() const {
      return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
    }

    bool is_contiguous() const {
      size_t expected_stride = 1;
      for (size_t i = shape.size(); i-- > 0; ) {
        if (strides[i] != expected_stride) return false;
        expected_stride *= shape[i];
      }
      return true;
    }
  };

  template<typename T>
    requires std::is_floating_point_v<T>
  class Parameter { 
    public:
      using is_param_like = std::true_type;
      using is_quantized = std::false_type;
      using DataType = T;

      explicit Parameter(const std::initializer_list<size_t>& shape) 
        : Parameter(std::vector<size_t>(shape.begin(), shape.end())) {}

      template<typename S>
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

      void view(const std::initializer_list<size_t>& new_shape) {
        std::vector<size_t> new_shape_vec(new_shape);
        size_t new_numel = std::accumulate(new_shape_vec.begin(), new_shape_vec.end(), size_t{1}, std::multiplies<size_t>{});
        assert(new_numel == data_.size() && "Total number of elements must remain the same when reshaping");

        shape_ = std::move(new_shape_vec);
        strides_.resize(shape_.size());
        std::exclusive_scan(
          shape_.rbegin(), shape_.rend(), strides_.rbegin(), size_t{1}, std::multiplies<size_t>{}
        );
      }

      std::vector<T>& grad() { return grad_; }
      const std::vector<T>& grad() const { return grad_; }

      std::vector<T>& data() { return data_; }
      const std::vector<T>& data() const { return data_; }

      const std::vector<size_t>& size() const { return shape_; }
      size_t size(size_t i) const { return shape_[i]; }

      size_t rank() const { return shape_.size(); }
      size_t numel() const { return data_.size(); }

      template<typename... Args>
        requires (std::convertible_to<Args, Slice> && ...)
      View<T> operator[](Args... args) {
        std::vector<Slice> slices{args...};
        assert((slices.size() <= shape_.size()) && "There cannot be more slices than dimensions");
        while (slices.size() < shape_.size()) {
          slices.emplace_back();
        }

        auto params = detail::compute_view(slices, shape_, strides_);
        return View<T>{
          .ref = *this,
          .offset = params.offset,
          .shape = std::move(params.shape),
          .strides = std::move(params.strides),
        };
      }

      template<typename... Args>
        requires (std::convertible_to<Args, Slice> && ...)
      const View<const T> operator[](Args... args) const {
        std::vector<Slice> slices{args...};
        assert((slices.size() <= shape_.size()) && "There cannot be more slices than dimensions");
        while (slices.size() < shape_.size()) {
          slices.emplace_back();
        }

        auto params = detail::compute_view(slices, shape_, strides_);
        return View<const T>{
          .ref = *this,
          .offset = params.offset,
          .shape = std::move(params.shape),
          .strides = std::move(params.strides),
        };
      }

      template<typename S>
        requires detail::NestedSpan<S, T>
      void fill(const S& new_data) {
        auto new_shape = detail::deduce_shape(new_data);
        assert(new_shape == shape_ && "Cannot fill parameter with data of different shape");

        detail::fill(new_data, new_shape, [&](const auto& leaf, size_t offset) {
          std::copy(leaf.begin(), leaf.end(), data_.begin() + offset);
        });
      }

      void zero_grad() {
        std::fill(grad_.begin(), grad_.end(), T(0));
      }

      void accumulate(const std::vector<T>& new_grad) {
        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= grad_.size(); i += vec_size * unroll_factor) {
          simd::unroll<unroll_factor>([&]<size_t index>() {
            constexpr size_t offset = index * vec_size;

            auto grad_vec = simd::load<T>(&grad_[i + offset]);
            auto new_vec = simd::load<T>(&new_grad[i + offset]);
            grad_vec = simd::add(grad_vec, new_vec);
            simd::store(&grad_[i + offset], grad_vec);
          });
        }

        for (; i < grad_.size(); ++i) {
          grad_[i] += new_grad[i];
        }
      }

      void update(const std::vector<T>& new_val) {
        data_ = new_val;
      }

      virtual void save_to_bin(const std::string& path_str, bool include_metadata = true) const {
        std::filesystem::path path(path_str + ".bin");
        std::ofstream out(path, std::ios::binary);

        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        if (include_metadata) {
          size_t rank = shape_.size();
          out.write(dtype_name(), std::strlen(dtype_name()) + 1);
          out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
          out.write(reinterpret_cast<const char*>(shape_.data()), shape_.size() * sizeof(size_t));
        }
        out.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(T));
      }

    protected:
      std::vector<size_t> shape_;
      std::vector<size_t> strides_;
      std::vector<T> data_;
      std::vector<T> grad_;

      explicit Parameter(std::vector<size_t>&& shape) : shape_(std::move(shape)) {
        strides_.resize(shape_.size());
        std::exclusive_scan(
          shape_.rbegin(), shape_.rend(), strides_.rbegin(), size_t{1}, std::multiplies<size_t>{}
        );

        size_t num = std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>{});
        data_.resize(num);
        grad_.resize(num);
      }

      constexpr const char* dtype_name() const {
        if constexpr (std::is_same_v<T, float>) return "float32\0";
        else if constexpr (std::is_same_v<T, double>) return "float64\0";
        else return "unknown\0";
      }
  };

  template<typename Q, typename T = float>
    requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
  class Quantized : public Parameter<T> {
    public:
      using is_param_like = std::true_type;
      using is_quantized = std::true_type;
      using QuantizedType = Q;

      explicit Quantized(const std::initializer_list<size_t>& shape, float scale = 1.0f, float zero_point = 0.0f) 
        : Parameter<T>(std::vector<size_t>(shape.begin(), shape.end())), scale_(scale), zero_point_(zero_point) {}

      template<typename S>
        requires detail::NestedSpan<S, Q>
      explicit Quantized(const S& span, float scale = 1.0f, float zero_point = 0.0f)
        : Parameter<T>(detail::deduce_shape(span)), scale_(scale), zero_point_(zero_point) {
          detail::fill(span, this->shape_, [&](const auto& leaf, size_t offset) {
            T scale_cast = static_cast<T>(scale);
            T zero_point_cast = static_cast<T>(zero_point);

            constexpr size_t q_vec_size = simd::vec<Q>::size;
            constexpr size_t f_vec_size = simd::vec<T>::size;
            constexpr size_t num_slices = q_vec_size / f_vec_size;
            constexpr size_t unroll_factor = std::min(num_slices, 4UL);

            int i = 0;
            for (; i + q_vec_size <= leaf.size(); i += q_vec_size) {
              auto scale_vec = simd::set1<T>(scale_cast);
              auto zero_point_vec = simd::set1<T>(zero_point_cast);

              auto q_vec = simd::load<Q>(&leaf[i]);
              int j = 0;
              for (; j + unroll_factor <= num_slices; j += unroll_factor) {
                simd::unroll<unroll_factor>([&]<size_t slice>() {
                  constexpr size_t offset0 = slice * f_vec_size;

                  auto q_float_vec = simd::cast<T, slice>(q_vec);

                  auto val_vec = simd::sub(q_float_vec, zero_point_vec);
                  val_vec = simd::mul(val_vec, scale_vec);

                  simd::store(&this->data_[i + j + offset + offset0], val_vec);
                });
              }
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

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
          simd::unroll<unroll_factor>([&]<size_t index>() {
            constexpr size_t offset = index * vec_size;

            auto val_vec = simd::load<T>(&vals[i + offset]);
            auto inv_scale_vec = simd::set1<T>(inv_scale);
            auto zero_point_vec = simd::set1<T>(zero_point_cast);
            auto q_vec = simd::fmadd(val_vec, inv_scale_vec, zero_point_vec);

            simd::store(&quantized_data[i + offset], simd::cast<Q>(q_vec));
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

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
          simd::unroll<unroll_factor>([&]<size_t index>() {
            constexpr size_t offset = index * vec_size;

            auto val_vec = simd::load<T>(&vals[i + offset]);
            auto scale_vec = simd::set1<T>(scale_);
            auto inv_scale_vec = simd::set1<T>(inv_scale);
            auto zero_point_vec = simd::set1<T>(zero_point_cast);

            auto q_vec = simd::fmadd(val_vec, inv_scale_vec, zero_point_vec);
            q_vec = simd::round(q_vec);

            auto dq_vec = simd::sub(q_vec, zero_point_vec);
            dq_vec = simd::mul(dq_vec, scale_vec);
            simd::store(&fake_quantized_data[i + offset], dq_vec);
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

      void save_to_bin(const std::string& path_str, bool dequantize = false, bool include_metadata = true) const {
        std::filesystem::path path(path_str + ".bin");
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        if (include_metadata) {
          size_t rank = this->shape_.size();
          out.write(qtype_name(), std::strlen(qtype_name()) + 1);
          out.write(this->dtype_name(), std::strlen(this->dtype_name()) + 1);
          out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
          out.write(reinterpret_cast<const char*>(this->shape_.data()), this->shape_.size() * sizeof(size_t));
          out.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
          out.write(reinterpret_cast<const char*>(&zero_point_), sizeof(zero_point_));
        }
        if (dequantize) {
          std::vector<T> data = fake_quantized();
          out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        } else {
          std::vector<Q> data = quantized();
          out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(Q));
        }
      }

    private:
      float scale_ = 1.0f;
      float zero_point_ = 0.0f;

      constexpr const char* qtype_name() const {
        if constexpr (std::is_same_v<Q, int8_t>) return "int8\0";
        else if constexpr (std::is_same_v<Q, int16_t>) return "int16\0";
        else return "unknown\0";
      }
  };

  template<typename T>
  concept ParamLike = T::is_param_like::value;

  template<typename T>
  using RefVec = std::vector<std::reference_wrapper<T>>;

  template<typename DedupedTuple>
  struct ParameterPack {
    detail::TransformTuple_t<RefVec, DedupedTuple> data{};

    template<typename... Ts>
      requires (std::derived_from<Ts, Parameter<typename Ts::DataType>> && ...)
    ParameterPack(Ts&... params) {
      (std::get<RefVec<Ts>>(data).emplace_back(params), ...);
    }

    template<typename T>
      requires std::derived_from<T, Parameter<typename T::DataType>>
    void add_parameter(T& param) {
      std::get<RefVec<T>>(data).emplace_back(param);
    }
  };
  template<typename... Ts>
  ParameterPack(Ts&...) -> ParameterPack<detail::Canonicalized_t<std::decay_t<Ts>...>>;

  template<typename T, typename IsQuantized = std::false_type>
  struct TaggedVector : std::vector<T> {
    using std::vector<T>::vector;
    using std::vector<T>::operator=;

    TaggedVector(std::vector<T>&& v) : std::vector<T>(std::move(v)) {}
    TaggedVector(const std::vector<T>& v) : std::vector<T>(v) {}
  };

  template<typename T>
  struct ExtractType {};
  template<typename T>
  struct ExtractType<Parameter<T>> { using Type = TaggedVector<T, std::false_type>; };
  template<typename Q, typename T>
  struct ExtractType<Quantized<Q, T>> { using Type = TaggedVector<T, std::true_type>; };
  template<typename T>
  using ExtractType_t = typename ExtractType<T>::Type;

  template<typename DedupedTuple>
  using ExtractedVector = detail::TransformTuple_t<ExtractType_t, DedupedTuple>;
}