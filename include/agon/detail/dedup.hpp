#pragma once

#include <tuple>
#include <cstdint>

namespace agon::detail {
  template <typename T, typename... Ts>
  concept ContainedInPack = (std::same_as<T, Ts> || ...);

  template <typename T, typename... Ts>
  struct AppendIfNotInPack {
    using Type = std::conditional_t<ContainedInPack<T, Ts...>, std::tuple<Ts...>, std::tuple<Ts..., T>>;
  };
  template <typename T, typename... Ts>
  using AppendIfNotInPack_t = AppendIfNotInPack<T, Ts...>::Type;

  template <template <typename...> typename Trait, typename Tuple>
  struct ApplyPackTraitWithTuple {};
  template <template <typename...> typename Trait, typename... Ts>
  struct ApplyPackTraitWithTuple<Trait, std::tuple<Ts...>> {
    using Type = Trait<Ts...>;
  };
  template <template <typename...> typename Trait, typename Tuple>
  using ApplyPackTraitWithTuple_t = ApplyPackTraitWithTuple<Trait, Tuple>::Type;

  template <typename T, typename Tuple>
  using PrependToTuple_t = decltype(
    std::tuple_cat(std::declval<std::tuple<T>>(), std::declval<Tuple>())
  );

  template <typename T, typename... Ts>
  struct DeduplicatedPack {
    using Type = ApplyPackTraitWithTuple_t<AppendIfNotInPack_t, PrependToTuple_t<T, typename DeduplicatedPack<Ts...>::Type>>;
  };
  template <typename T>
  struct DeduplicatedPack<T> {
    using Type = std::tuple<T>;
  };
  template <typename... Ts>
  using DeduplicatedPack_t = DeduplicatedPack<Ts...>::Type;

  template <typename T>
  struct TypeRank;

  template <> struct TypeRank<float> : std::integral_constant<int, 0> {};
  template <> struct TypeRank<double> : std::integral_constant<int, 1> {};
  template <> struct TypeRank<int8_t> : std::integral_constant<int, 2> {};
  template <> struct TypeRank<int16_t> : std::integral_constant<int, 3> {};

  template <typename T>
    requires requires { typename T::DataType; } && (!T::is_quantized::value)
  struct TypeRank<T>
      : std::integral_constant<int, TypeRank<typename T::DataType>::value * 10> {};

  template <typename T>
    requires requires { typename T::DataType; typename T::QuantizedType; } && T::is_quantized::value
  struct TypeRank<T>
      : std::integral_constant<int, TypeRank<typename T::DataType>::value * 10 + TypeRank<typename T::QuantizedType>::value + 1> {};

  template <typename T, typename SortedTuple>
  struct SortedInsert;
  template <typename T>
  struct SortedInsert<T, std::tuple<>> {
    using Type = std::tuple<T>;
  };
  template <typename T, typename Head, typename... Tail>
  struct SortedInsert<T, std::tuple<Head, Tail...>> {
    using Type = std::conditional_t<
      (TypeRank<T>::value <= TypeRank<Head>::value),
      std::tuple<T, Head, Tail...>,
      PrependToTuple_t<Head, typename SortedInsert<T, std::tuple<Tail...>>::Type>
    >;
  };
  template <typename T, typename SortedTuple>
  using SortedInsert_t = typename SortedInsert<T, SortedTuple>::Type;

  template <typename Tuple>
  struct SortTuple;
  template <>
  struct SortTuple<std::tuple<>> {
    using Type = std::tuple<>;
  };
  template <typename Head, typename... Tail>
  struct SortTuple<std::tuple<Head, Tail...>> {
    using Type = SortedInsert_t<Head, typename SortTuple<std::tuple<Tail...>>::Type>;
  };
  template <typename Tuple>
  using SortTuple_t = typename SortTuple<Tuple>::Type;

  template <typename... Ts>
  using Canonicalized_t = SortTuple_t<DeduplicatedPack_t<Ts...>>;

  template <template <typename...> typename Trait, typename Tuple>
  struct TransformTuple {};
  template <template <typename...> typename Trait, typename... Ts>
  struct TransformTuple<Trait, std::tuple<Ts...>> {
    using Type = std::tuple<Trait<Ts>...>;
  };
  template <template <typename...> typename Trait, typename Tuple>
  using TransformTuple_t = TransformTuple<Trait, Tuple>::Type;
}