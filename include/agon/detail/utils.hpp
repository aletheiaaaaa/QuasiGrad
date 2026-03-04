#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <eve/wide.hpp>

#include "arch.hpp"

namespace agon::simd {
  template<typename F, typename T>
  concept IsUpcast = eve::wide<F>::size() < eve::wide<T>::size();

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
    } else if (std::is_same_v<T, int32_t>) {
      func.template operator()<int32_t>();
    } else {
      throw std::runtime_error("Unsupported data type for SIMD operation");
    }
  }
}