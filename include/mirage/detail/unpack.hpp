#pragma once

#include <cassert>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace mirage::detail {
template <typename T>
struct is_span : std::false_type {};

template <typename T, int E>
struct is_span<std::span<T, E>> : std::true_type {};

template <typename T>
struct span_scalar {
  using type = T;
};

template <typename T, int E>
struct span_scalar<std::span<T, E>> : span_scalar<T> {};

template <typename S, typename T>
concept NestedSpan = is_span<S>::value && std::is_same_v<typename span_scalar<S>::type, T>;

template <typename S, typename F>
  requires is_span<S>::value
void walk(const S& span, std::vector<int>& shape, int& offset, bool first, F&& func) {
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

template <typename S>
  requires NestedSpan<S, typename span_scalar<S>::type>
std::vector<int> deduce_shape(const S& span) {
  std::vector<int> shape;
  int dummy = 0;
  walk(span, shape, dummy, true, [](auto&, int) {});

  return shape;
}

template <typename S, typename F>
  requires NestedSpan<S, typename span_scalar<S>::type>
void fill(const S& span, std::vector<int>& shape, F&& func) {
  int offset = 0;
  walk(span, shape, offset, false, func);
}

template <typename S, typename Out, typename F>
  requires NestedSpan<S, typename span_scalar<S>::type>
void unpack(
  const S& span,
  std::vector<int>& shape,
  std::vector<Out>& data,
  std::vector<int>& strides,
  F&& func
) {
  shape = deduce_shape(span);

  int numel = std::accumulate(shape.begin(), shape.end(), int{1}, std::multiplies<int>{});
  data.resize(numel);

  fill(span, shape, [&](const auto& leaf, int off) { func(leaf, data, off); });

  strides.resize(shape.size());
  std::exclusive_scan(
    shape.rbegin(), shape.rend(), strides.rbegin(), int{1}, std::multiplies<int>{}
  );
}

template <typename S, typename T>
  requires NestedSpan<S, T>
void unpack(
  const S& span, std::vector<int>& shape, std::vector<T>& data, std::vector<int>& strides
) {
  unpack(span, shape, data, strides, [](const auto& leaf, std::vector<T>& out, int off) {
    std::copy(leaf.begin(), leaf.end(), out.begin() + off);
  });
}
}  // namespace mirage::detail