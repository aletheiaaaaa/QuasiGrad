#pragma once

#include "../arch.h"
#include "../types.h"

#include <cmath>
#if defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__SSE4_1__)
  #include <smmintrin.h>
#endif

namespace agon::simd {
#if defined(__AVX512F__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> round(Vec<CURRENT_ARCH, T> v) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_roundscale_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_roundscale_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
  }
#elif defined(__AVX2__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> round(Vec<CURRENT_ARCH, T> v) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_round_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_round_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
  }
#elif defined(__SSE4_1__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> round(Vec<CURRENT_ARCH, T> v) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_round_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_round_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
  }
#else
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> round(Vec<CURRENT_ARCH, T> v) {
    return Vec<CURRENT_ARCH, T>(std::round(v.data));
  }
#endif
}
