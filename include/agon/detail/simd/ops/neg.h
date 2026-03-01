#pragma once

#include "../arch.h"
#include "../types.h"
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
  inline Vec<CURRENT_ARCH, T> neg(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_xor_ps(a.data, _mm512_set1_ps(-0.0f)));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_xor_pd(a.data, _mm512_set1_pd(-0.0)));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> neg(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_xor_ps(a.data, _mm256_set1_ps(-0.0f)));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_xor_pd(a.data, _mm256_set1_pd(-0.0)));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> neg(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_xor_ps(a.data, _mm_set1_ps(-0.0f)));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_xor_pd(a.data, _mm_set1_pd(-0.0)));
    }
  }

#else
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> neg(const Vec<CURRENT_ARCH, T>& a) {
    return Vec<CURRENT_ARCH, T>(-a.data);
  }

#endif

}
