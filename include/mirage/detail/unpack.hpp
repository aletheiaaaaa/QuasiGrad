#pragma once

#include <cassert>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace mirage::detail {
  template<typename T>
  struct is_span : std::false_type {};

  template<typename T, size_t E>
  struct is_span<std::span<T, E>> : std::true_type {};

  template<typename T>
  struct span_scalar { using type = T; };

  template<typename T, size_t E>
  struct span_scalar<std::span<T, E>> : span_scalar<T> {};

  template<typename S, typename T>
  concept NestedSpan = is_span<S>::value
    && std::is_same_v<typename span_scalar<S>::type, T>;

  template<typename S, typename F>
    requires is_span<S>::value
  void walk(const S& span, std::vector<size_t>& shape, size_t& offset, bool first, F&& func) {
    if (first) shape.push_back(span.size());

    if constexpr (is_span<typename S::element_type>::value) {
      for (const auto& sub : span) {
        assert(sub.size() == span[0].size() && "Inconsistent inner dimensions in nested span");
        walk(sub, shape, offset, first && (&sub == &span[0]), func);
      }
    } else {
      func(span, offset);
      offset += span.size();
    }
  }

  template<typename S>
    requires NestedSpan<S, typename span_scalar<S>::type>
  std::vector<size_t> deduce_shape(const S& span) {
    std::vector<size_t> shape;
    size_t dummy = 0;
    walk(span, shape, dummy, true, [](auto&, size_t) {});

    return shape;
  }

  template<typename S, typename F>
    requires NestedSpan<S, typename span_scalar<S>::type>
  void fill(const S& span, std::vector<size_t>& shape, F&& func) {
    size_t offset = 0;
    walk(span, shape, offset, false, func);
  }

  template<typename S, typename Out, typename F>
    requires NestedSpan<S, typename span_scalar<S>::type>
  void unpack(
    const S& span, 
    std::vector<size_t>& shape, 
    std::vector<Out>& data, 
    std::vector<size_t>& strides, 
    F&& func
  ) {
    shape = deduce_shape(span);

    size_t numel = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
    data.resize(numel);

    fill(span, shape, [&](const auto& leaf, size_t off) {
      func(leaf, data, off);
    });

    strides.resize(shape.size());
    std::exclusive_scan(shape.rbegin(), shape.rend(), strides.rbegin(), size_t{1}, std::multiplies<size_t>{});
  }

  template<typename S, typename T>
    requires NestedSpan<S, T>
  void unpack(
    const S& span, 
    std::vector<size_t>& shape, 
    std::vector<T>& data,
    std::vector<size_t>& strides
  ) {
    unpack(span, shape, data, strides,
      [](const auto& leaf, std::vector<T>& out, size_t off) {
        std::copy(leaf.begin(), leaf.end(), out.begin() + off);
      }
    );
  }
}