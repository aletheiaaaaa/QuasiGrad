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
  inline void store(T* ptr, const Vec<CURRENT_ARCH, T>& v) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
    } else if constexpr (std::is_same_v<T, float>) {
      _mm512_storeu_ps(ptr, v.data);
    } else if constexpr (std::is_same_v<T, double>) {
      _mm512_storeu_pd(ptr, v.data);
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline void store(T* ptr, const Vec<CURRENT_ARCH, T>& v) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
    } else if constexpr (std::is_same_v<T, float>) {
      _mm256_storeu_ps(ptr, v.data);
    } else if constexpr (std::is_same_v<T, double>) {
      _mm256_storeu_pd(ptr, v.data);
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline void store(T* ptr, const Vec<CURRENT_ARCH, T>& v) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
    } else if constexpr (std::is_same_v<T, float>) {
      _mm_storeu_ps(ptr, v.data);
    } else if constexpr (std::is_same_v<T, double>) {
      _mm_storeu_pd(ptr, v.data);
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline void store(T* ptr, const Vec<CURRENT_ARCH, T>& v) {
    *ptr = v.data;
  }

#endif
}
