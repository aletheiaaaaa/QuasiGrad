#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <eve/wide.hpp>

#include "arch.hpp"

namespace mirage::detail {
  template<typename T>
  struct TypeName {
    static std::string name() { return "unknown"; }
  };
  template<>
  struct TypeName<float> {
    static std::string name() { return "float"; }
  };
  template<>
  struct TypeName<double> {
    static std::string name() { return "double"; }
  };
  template<>
  struct TypeName<int16_t> {
    static std::string name() { return "int16_t"; }
  };
  template<>
  struct TypeName<int8_t> {
    static std::string name() { return "int8_t"; }
  };

  template<typename F, typename T>
  concept IsUpcast = eve::wide<F>::size() < eve::wide<T>::size();

  constexpr int UNROLL_FACTOR = 
    (CURRENT_ARCH == Arch::AVX512 || CURRENT_ARCH == Arch::NEON) ? 4 :
    (CURRENT_ARCH == Arch::AVX2) ? 2 : 1;

  constexpr int OUTER_REGS = UNROLL_FACTOR * 2;
  constexpr int INNER_REGS = std::max(1, UNROLL_FACTOR / 2);

  template<size_t N, typename F>
  constexpr void unroll(F&& func) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      (func.template operator()<Is>(), ...);
    }(std::make_index_sequence<N>{});
  }
}