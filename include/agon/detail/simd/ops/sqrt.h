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
  inline Vec<CURRENT_ARCH, T> sqrt(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_sqrt_ps(a.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_sqrt_pd(a.data));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> sqrt(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_sqrt_ps(a.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_sqrt_pd(a.data));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> sqrt(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_sqrt_ps(a.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_sqrt_pd(a.data));
    }
  }

#else
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> sqrt(const Vec<CURRENT_ARCH, T>& a) {
    return Vec<CURRENT_ARCH, T>(std::sqrt(a.data));
  }

#endif
}
