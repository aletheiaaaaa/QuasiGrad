#pragma once

#include "../arch.h"
#include "../types.h"

#include <cmath>
#include <type_traits>
#if defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__SSE4_1__)
  #include <smmintrin.h>
#endif

namespace agon::simd {
  template<typename T, size_t N = 0, typename F>
  inline Vec<CURRENT_ARCH, T> cast(Vec<CURRENT_ARCH, F> v);

#if defined(__AVX512F__)

  template<typename T, size_t N, typename F>
    requires IsOperable<T> && IsOperable<F>
  inline Vec<CURRENT_ARCH, T> cast(Vec<CURRENT_ARCH, F> v) {

    if constexpr (std::is_same_v<F, int8_t>) {
      if constexpr (std::is_same_v<T, int16_t>) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return Vec<CURRENT_ARCH, T>(_mm512_cvtepi8_epi16(chunk));

      } else if constexpr (std::is_same_v<T, float>) {
        __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
        return Vec<CURRENT_ARCH, T>(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(chunk)));

      } else if constexpr (std::is_same_v<T, double>) {
        __m128i lane = _mm512_extracti32x4_epi32(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        __m512i i64 = _mm512_cvtepi8_epi64(chunk);
        return Vec<CURRENT_ARCH, T>(_mm512_cvtepi64_pd(i64));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int8_t on AVX512");
      }
    }

    else if constexpr (std::is_same_v<F, int16_t>) {
      if constexpr (std::is_same_v<T, float>) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return Vec<CURRENT_ARCH, T>(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(chunk)));

      } else if constexpr (std::is_same_v<T, double>) {
        __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
        __m512i i64 = _mm512_cvtepi16_epi64(chunk);
        return Vec<CURRENT_ARCH, T>(_mm512_cvtepi64_pd(i64));

      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m128i narrow = _mm512_cvtepi16_epi8(v.data);
        return Vec<CURRENT_ARCH, T>(_mm512_castsi128_si512(narrow));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int16_t on AVX512");
      }
    }

    else if constexpr (std::is_same_v<F, float>) {
      if constexpr (std::is_same_v<T, double>) {
        __m256 chunk = (N == 0) ? _mm512_castps512_ps256(v.data) : _mm512_extractf32x8_ps(v.data, 1);
        return Vec<CURRENT_ARCH, T>(_mm512_cvtps_pd(chunk));

      } else if constexpr (std::is_same_v<T, int16_t>) {
        __m512i i32 = _mm512_cvtps_epi32(v.data);
        __m256i narrow = _mm512_cvtepi32_epi16(i32);
        return Vec<CURRENT_ARCH, T>(_mm512_castsi256_si512(narrow));

      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m512i i32 = _mm512_cvtps_epi32(v.data);
        __m128i narrow = _mm512_cvtepi32_epi8(i32);
        return Vec<CURRENT_ARCH, T>(_mm512_castsi128_si512(narrow));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from float on AVX512");
      }
    }

    else if constexpr (std::is_same_v<F, double>) {
      if constexpr (std::is_same_v<T, float>) {
        __m256 narrow = _mm512_cvtpd_ps(v.data);
        return Vec<CURRENT_ARCH, T>(_mm512_castps256_ps512(narrow));

      } else if constexpr (std::is_same_v<T, int16_t>) {
        __m256i i32 = _mm512_cvtpd_epi32(v.data);
        __m128i narrow = _mm256_cvtepi32_epi16(i32);
        return Vec<CURRENT_ARCH, T>(_mm512_castsi128_si512(narrow));

      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m256i i32 = _mm512_cvtpd_epi32(v.data);
        __m128i narrow = _mm256_cvtepi32_epi8(i32);
        return Vec<CURRENT_ARCH, T>(_mm512_castsi128_si512(narrow));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from double on AVX512");
      }
    }
    else {
      static_assert(std::is_same_v<F, void>, "Unsupported source type for cast on AVX512");
    }
  }

#elif defined(__AVX2__)

  template<typename T, size_t N, typename F>
    requires IsOperable<T> && IsOperable<F>
  inline Vec<CURRENT_ARCH, T> cast(Vec<CURRENT_ARCH, F> v) {

    if constexpr (std::is_same_v<F, int8_t>) {
      if constexpr (std::is_same_v<T, int16_t>) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return Vec<CURRENT_ARCH, T>(_mm256_cvtepi8_epi16(chunk));

      } else if constexpr (std::is_same_v<T, float>) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        return Vec<CURRENT_ARCH, T>(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(chunk)));

      } else if constexpr (std::is_same_v<T, double>) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 4);
        __m128i chunk = _mm_srli_si128(lane, (N % 4) * 4);
        __m256i i64 = _mm256_cvtepi8_epi64(chunk);
        return Vec<CURRENT_ARCH, T>(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i64)));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int8_t on AVX2");
      }
    }

    else if constexpr (std::is_same_v<F, int16_t>) {
      if constexpr (std::is_same_v<T, float>) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return Vec<CURRENT_ARCH, T>(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(chunk)));

      } else if constexpr (std::is_same_v<T, double>) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        __m256i i64 = _mm256_cvtepi16_epi64(chunk);
        return Vec<CURRENT_ARCH, T>(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i64)));

      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m256i shuf = _mm256_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0,
          -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0
        );
        __m256i shuffled = _mm256_shuffle_epi8(v.data, shuf);
        return Vec<CURRENT_ARCH, T>(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int16_t on AVX2");
      }
    }

    else if constexpr (std::is_same_v<F, float>) {
      if constexpr (std::is_same_v<T, double>) {
        __m128 chunk = (N == 0) ? _mm256_castps256_ps128(v.data) : _mm256_extractf128_ps(v.data, 1);
        return Vec<CURRENT_ARCH, T>(_mm256_cvtps_pd(chunk));

      } else if constexpr (std::is_same_v<T, int16_t>) {
        __m256i i32 = _mm256_cvtps_epi32(v.data);
        __m256i shuf = _mm256_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
          -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        __m256i shuffled = _mm256_shuffle_epi8(i32, shuf);
        return Vec<CURRENT_ARCH, T>(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));

      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m256i i32 = _mm256_cvtps_epi32(v.data);
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        lo = _mm_shuffle_epi8(lo, shuf);
        hi = _mm_shuffle_epi8(hi, shuf);
        return Vec<CURRENT_ARCH, T>(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from float on AVX2");
      }
    }

    else if constexpr (std::is_same_v<F, double>) {
      if constexpr (std::is_same_v<T, float>) {
        __m128 narrow = _mm256_cvtpd_ps(v.data);
        return Vec<CURRENT_ARCH, T>(_mm256_castps128_ps256(narrow));

      } else if constexpr (std::is_same_v<T, int16_t>) {
        __m128i i32 = _mm256_cvtpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm256_castsi128_si256(_mm_shuffle_epi8(i32, shuf)));

      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m128i i32 = _mm256_cvtpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm256_castsi128_si256(_mm_shuffle_epi8(i32, shuf)));

      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from double on AVX2");
      }
    }
    else {
      static_assert(std::is_same_v<F, void>, "Unsupported source type for cast on AVX2");
    }
  }

#elif defined(__SSE4_1__)

  template<typename T, size_t N, typename F>
    requires IsOperable<T> && IsOperable<F>
  inline Vec<CURRENT_ARCH, T> cast(Vec<CURRENT_ARCH, F> v) {

    if constexpr (std::is_same_v<F, int8_t>) {
      if constexpr (std::is_same_v<T, int16_t>) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return Vec<CURRENT_ARCH, T>(_mm_cvtepi8_epi16(chunk));
      } else if constexpr (std::is_same_v<T, float>) {
        __m128i chunk = _mm_srli_si128(v.data, N * 4);
        return Vec<CURRENT_ARCH, T>(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(chunk)));
      } else if constexpr (std::is_same_v<T, double>) {
        __m128i chunk = _mm_srli_si128(v.data, N * 2);
        return Vec<CURRENT_ARCH, T>(_mm_cvtepi32_pd(_mm_cvtepi8_epi32(chunk)));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int8_t on SSE4_1");
      }
    }

    else if constexpr (std::is_same_v<F, int16_t>) {
      if constexpr (std::is_same_v<T, float>) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return Vec<CURRENT_ARCH, T>(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(chunk)));
      } else if constexpr (std::is_same_v<T, double>) {
        __m128i chunk = _mm_srli_si128(v.data, N * 4);
        return Vec<CURRENT_ARCH, T>(_mm_cvtepi32_pd(_mm_cvtepi16_epi32(chunk)));
      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm_shuffle_epi8(v.data, shuf));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int16_t on SSE4_1");
      }
    }

    else if constexpr (std::is_same_v<F, float>) {
      if constexpr (std::is_same_v<T, double>) {
        __m128 chunk = (N == 0) ? v.data : _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v.data), 8));
        return Vec<CURRENT_ARCH, T>(_mm_cvtps_pd(chunk));
      } else if constexpr (std::is_same_v<T, int16_t>) {
        __m128i i32 = _mm_cvtps_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm_shuffle_epi8(i32, shuf));
      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m128i i32 = _mm_cvtps_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm_shuffle_epi8(i32, shuf));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from float on SSE4_1");
      }
    }

    else if constexpr (std::is_same_v<F, double>) {
      if constexpr (std::is_same_v<T, float>) {
        return Vec<CURRENT_ARCH, T>(_mm_cvtpd_ps(v.data));
      } else if constexpr (std::is_same_v<T, int16_t>) {
        __m128i i32 = _mm_cvtpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 4, 1, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm_shuffle_epi8(i32, shuf));
      } else if constexpr (std::is_same_v<T, int8_t>) {
        __m128i i32 = _mm_cvtpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0
        );
        return Vec<CURRENT_ARCH, T>(_mm_shuffle_epi8(i32, shuf));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from double on SSE4_1");
      }
    }
    else {
      static_assert(std::is_same_v<F, void>, "Unsupported source type for cast on SSE4_1");
    }
  }

#else
  template<typename T, size_t N, typename F>
    requires IsOperable<T> && IsOperable<F>
  inline Vec<CURRENT_ARCH, T> cast(Vec<CURRENT_ARCH, F> v) {

    if constexpr (std::is_same_v<F, int8_t>) {
      if constexpr (std::is_same_v<T, int16_t>) {
        return Vec<CURRENT_ARCH, T>(static_cast<int16_t>(v.data));
      } else if constexpr (std::is_same_v<T, float>) {
        return Vec<CURRENT_ARCH, T>(static_cast<float>(v.data));
      } else if constexpr (std::is_same_v<T, double>) {
        return Vec<CURRENT_ARCH, T>(static_cast<double>(v.data));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int8_t on GENERIC");
      }
    }
    else if constexpr (std::is_same_v<F, int16_t>) {
      if constexpr (std::is_same_v<T, float>) {
        return Vec<CURRENT_ARCH, T>(static_cast<float>(v.data));
      } else if constexpr (std::is_same_v<T, double>) {
        return Vec<CURRENT_ARCH, T>(static_cast<double>(v.data));
      } else if constexpr (std::is_same_v<T, int8_t>) {
        return Vec<CURRENT_ARCH, T>(static_cast<int8_t>(v.data));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from int16_t on GENERIC");
      }
    }
    else if constexpr (std::is_same_v<F, float>) {
      if constexpr (std::is_same_v<T, double>) {
        return Vec<CURRENT_ARCH, T>(static_cast<double>(v.data));
      } else if constexpr (std::is_same_v<T, int16_t>) {
        return Vec<CURRENT_ARCH, T>(static_cast<int16_t>(std::lround(v.data)));
      } else if constexpr (std::is_same_v<T, int8_t>) {
        return Vec<CURRENT_ARCH, T>(static_cast<int8_t>(std::lround(v.data)));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from float on GENERIC");
      }
    }
    else if constexpr (std::is_same_v<F, double>) {
      if constexpr (std::is_same_v<T, float>) {
        return Vec<CURRENT_ARCH, T>(static_cast<float>(v.data));
      } else if constexpr (std::is_same_v<T, int16_t>) {
        return Vec<CURRENT_ARCH, T>(static_cast<int16_t>(std::lround(v.data)));
      } else if constexpr (std::is_same_v<T, int8_t>) {
        return Vec<CURRENT_ARCH, T>(static_cast<int8_t>(std::lround(v.data)));
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported cast from double on GENERIC");
      }
    }
    else {
      static_assert(std::is_same_v<F, void>, "Unsupported source type for cast on GENERIC");
    }
  }

#endif
}
