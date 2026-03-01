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
  inline T setzero() {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_setzero_si512());
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm512_setzero_ps());
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm512_setzero_pd());
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline T setzero() {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_setzero_si256());
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_setzero_ps());
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_setzero_pd());
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline T setzero() {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_setzero_si128());
    } else if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_setzero_ps());
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_setzero_pd());
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline T setzero() {
    return Vec<CURRENT_ARCH, T>(T{});
  }
#endif
}
