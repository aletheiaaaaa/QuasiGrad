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
  inline Vec<CURRENT_ARCH, T> sign(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      __m512i ai = _mm512_castps_si512(a.data);
      __m512i sm = _mm512_castps_si512(_mm512_set1_ps(-0.0f));
      __m512i ob = _mm512_castps_si512(_mm512_set1_ps(1.0f));
      return Vec<CURRENT_ARCH, T>(_mm512_castsi512_ps(_mm512_or_si512(_mm512_and_si512(ai, sm), ob)));
    } else if constexpr (std::is_same_v<T, double>) {
      __m512i ai = _mm512_castpd_si512(a.data);
      __m512i sm = _mm512_castpd_si512(_mm512_set1_pd(-0.0));
      __m512i ob = _mm512_castpd_si512(_mm512_set1_pd(1.0));
      return Vec<CURRENT_ARCH, T>(_mm512_castsi512_pd(_mm512_or_si512(_mm512_and_si512(ai, sm), ob)));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      __m512i zero = _mm512_setzero_si512();
      return Vec<CURRENT_ARCH, T>(_mm512_or_si512(_mm512_movm_epi8(_mm512_cmpgt_epi8_mask(zero, a.data)),
                   _mm512_set1_epi8(1)));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm512_or_si512(_mm512_srai_epi16(a.data, 15), _mm512_set1_epi16(1)));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> sign(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm256_or_ps(_mm256_and_ps(a.data, _mm256_set1_ps(-0.0f)),
                  _mm256_set1_ps(1.0f)));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm256_or_pd(_mm256_and_pd(a.data, _mm256_set1_pd(-0.0)),
                  _mm256_set1_pd(1.0)));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_sign_epi8(_mm256_set1_epi8(1), a.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm256_sign_epi16(_mm256_set1_epi16(1), a.data));
    }
  }

#elif defined(__SSE4_1__)
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> sign(const Vec<CURRENT_ARCH, T>& a) {
    if constexpr (std::is_same_v<T, float>) {
      return Vec<CURRENT_ARCH, T>(_mm_or_ps(_mm_and_ps(a.data, _mm_set1_ps(-0.0f)), _mm_set1_ps(1.0f)));
    } else if constexpr (std::is_same_v<T, double>) {
      return Vec<CURRENT_ARCH, T>(_mm_or_pd(_mm_and_pd(a.data, _mm_set1_pd(-0.0)), _mm_set1_pd(1.0)));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_sign_epi8(_mm_set1_epi8(1), a.data));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return Vec<CURRENT_ARCH, T>(_mm_sign_epi16(_mm_set1_epi16(1), a.data));
    }
  }

#else
  template<typename T>
    requires IsOperable<T>
  inline Vec<CURRENT_ARCH, T> sign(const Vec<CURRENT_ARCH, T>& a) {
    return Vec<CURRENT_ARCH, T>(a.data > T(0) ? T(1) : a.data < T(0) ? T(-1) : T(0));
  }

#endif

}
