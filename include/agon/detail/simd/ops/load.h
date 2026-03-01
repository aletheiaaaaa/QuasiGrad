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
  inline Vec<CURRENT_ARCH, T> load(const T* ptr) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_loadu_ps(ptr));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_loadu_pd(ptr));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> load(const T* ptr) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_loadu_ps(ptr));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_loadu_pd(ptr));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> load(const T* ptr) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_loadu_ps(ptr));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_loadu_pd(ptr));
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> load(const T* ptr) {
    return Vec<CURRENT_ARCH, T>(*ptr);
  }

#endif
}
