#pragma once

#include "../arch.h"
#include "../types.h"

#include <algorithm>
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
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> min(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_min_epi8(a.data, b.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_min_epi16(a.data, b.data));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_min_ps(a.data, b.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_min_pd(a.data, b.data));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> min(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_min_epi8(a.data, b.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_min_epi16(a.data, b.data));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_min_ps(a.data, b.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_min_pd(a.data, b.data));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> min(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_min_epi8(a.data, b.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_min_epi16(a.data, b.data));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_min_ps(a.data, b.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_min_pd(a.data, b.data));
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> min(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    return Vec<CURRENT_ARCH, T>(std::min(a.data, b.data));
  }

#endif
}
