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
  inline Vec<CURRENT_ARCH, T> add(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_add_epi8(a.data, b.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_add_epi16(a.data, b.data));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_add_ps(a.data, b.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_add_pd(a.data, b.data));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> add(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_add_epi8(a.data, b.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_add_epi16(a.data, b.data));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_add_ps(a.data, b.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_add_pd(a.data, b.data));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> add(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_add_epi8(a.data, b.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_add_epi16(a.data, b.data));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_add_ps(a.data, b.data));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_add_pd(a.data, b.data));
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> add(const Vec<CURRENT_ARCH, T>& a, const Vec<CURRENT_ARCH, T>& b) {
    return Vec<CURRENT_ARCH, T>(a.data + b.data);
  }

#endif

}
