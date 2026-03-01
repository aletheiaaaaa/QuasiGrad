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
    requires IsOperable<T>
  inline T set1(T val) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_set1_epi8(val));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_set1_epi16(val));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_set1_ps(val));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_set1_pd(val));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline T set1(T val) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_set1_epi8(val));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_set1_epi16(val));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_set1_ps(val));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_set1_pd(val));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline T set1(T val) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_set1_epi8(val));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_set1_epi16(val));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_set1_ps(val));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_set1_pd(val));
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline T set1(T val) {
    return Vec<CURRENT_ARCH, T>(val);
  }
#endif
}
