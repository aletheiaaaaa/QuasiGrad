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
  inline Vec<CURRENT_ARCH, T> fnmsub(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b, const Vec<CURRENT_ARCH, T>& c) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_fnmsub_ps(a.data, b.data, c.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_fnmsub_pd(a.data, b.data, c.data));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> fnmsub(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b, const Vec<CURRENT_ARCH, T>& c) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_fnmsub_ps(a.data, b.data, c.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_fnmsub_pd(a.data, b.data, c.data));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> fnmsub(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b, const Vec<CURRENT_ARCH, T>& c) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_sub_ps(_mm_sub_ps(_mm_setzero_ps(), _mm_mul_ps(a.data, b.data)), c.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_sub_pd(_mm_sub_pd(_mm_setzero_pd(), _mm_mul_pd(a.data, b.data)), c.data));
    }
  }

#else
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> fnmsub(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b, const Vec<CURRENT_ARCH, T>& c) {
    return Vec<CURRENT_ARCH, T>(-(a.data * b.data) - c.data);
  }

#endif
}
