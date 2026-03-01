#pragma once

#include <cstddef>
#include <cstdint>
#include <concepts>
#include <stdexcept>
#include <typeinfo>

#include "arch.h"
#include "types.h"

namespace agon::simd {
  template<typename F, typename T>
  concept IsUpcast = Vec<CURRENT_ARCH, F>::size < simd::Vec<CURRENT_ARCH, T>::size;

  constexpr int UNROLL_FACTOR = 
    (CURRENT_ARCH == Arch::AVX512) ? 4 :
    (CURRENT_ARCH == Arch::AVX2) ? 2 : 1;

  template<size_t N, typename F>
  constexpr void unroll(F&& func) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      (func.template operator()<Is>(), ...);
    }(std::make_index_sequence<N>{});
  }

  template<typename T, typename F>
  constexpr void dispatch(F&& func) {
    if (std::is_same_v<T, float>) {
      func.template operator()<float>();
    } else if (std::is_same_v<T, double>) {
      func.template operator()<double>();
    } else {
      throw std::runtime_error("Unsupported data type for SIMD operation");
    }
  }
}