#pragma once

#include <tuple>

namespace mirage::detail {
  template<typename T, typename... Ts>
  concept ContainedInPack = (std::same_as<T, Ts> || ...);

  template<typename T, typename... Ts>
  struct AppendIfNotInPack {
    using Type = std::conditional_t<ContainedInPack<T, Ts...>, std::tuple<Ts...>, std::tuple<Ts..., T>>;
  };
  template<typename T, typename... Ts>
  using AppendIfNotInPack_t = AppendIfNotInPack<T, Ts...>::Type;

  template<template <typename...> typename Trait, typename Tuple>
  struct ApplyPackTraitWithTuple {};
  template<template <typename...> typename Trait, typename... Ts>
  struct ApplyPackTraitWithTuple<Trait, std::tuple<Ts...>> {
    using Type = Trait<Ts...>;
  };
  template<template <typename...> typename Trait, typename Tuple>
  using ApplyPackTraitWithTuple_t = ApplyPackTraitWithTuple<Trait, Tuple>::Type;

  template<typename T, typename Tuple>
  using PrependToTuple_t = decltype(
    std::tuple_cat(std::declval<std::tuple<T>>(), std::declval<Tuple>())
  );

  template<typename T, typename... Ts>
  struct DeduplicatedPack {
    using Type = ApplyPackTraitWithTuple_t<AppendIfNotInPack_t, PrependToTuple_t<T, typename DeduplicatedPack<Ts...>::Type>>;
  };
  template<typename T>
  struct DeduplicatedPack<T> {
    using Type = std::tuple<T>;
  };
  template<typename... Ts>
  using DeduplicatedPack_t = DeduplicatedPack<Ts...>::Type;

  template<template <typename...> typename Trait, typename Tuple>
  struct TransformTuple {};
  template<template <typename...> typename Trait, typename... Ts>
  struct TransformTuple<Trait, std::tuple<Ts...>> {
    using Type = std::tuple<Trait<Ts>...>;
  };
  template<template <typename...> typename Trait, typename Tuple>
  using TransformTuple_t = TransformTuple<Trait, Tuple>::Type;
}