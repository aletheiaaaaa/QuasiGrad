#pragma once

#include "../arch.h"
#include "../types.h"

#include <cmath>
#include <concepts>
#include <stdfloat>

#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
#endif

namespace agon::simd {
    template<typename T, size_t N = 0, typename F>
    inline T cast_trunc(F v);

#if defined(__AVX512F__)
    template<size_t N>
    inline VecI16<Arch::AVX512> cast_trunc(VecI8<Arch::AVX512> v) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return VecI16<Arch::AVX512>(_mm512_cvtepi8_epi16(chunk));
    }

    template<size_t N>
    inline VecI32<Arch::AVX512> cast_trunc(VecI8<Arch::AVX512> v) {
        __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
        return VecI32<Arch::AVX512>(_mm512_cvtepi8_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::AVX512> cast_trunc(VecI8<Arch::AVX512> v) {
        __m128i lane = _mm512_extracti32x4_epi32(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        return VecI64<Arch::AVX512>(_mm512_cvtepi8_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::AVX512> cast_trunc(VecI8<Arch::AVX512> v) {
        __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
        return VecF32<Arch::AVX512>(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(chunk)));
    }

    template<size_t N>
    inline VecF64<Arch::AVX512> cast_trunc(VecI8<Arch::AVX512> v) {
        __m128i lane = _mm512_extracti32x4_epi32(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        __m512i i64 = _mm512_cvtepi8_epi64(chunk);
        return VecF64<Arch::AVX512>(_mm512_cvtepi64_pd(i64));
    }

    template<size_t N>
    inline VecI32<Arch::AVX512> cast_trunc(VecI16<Arch::AVX512> v) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return VecI32<Arch::AVX512>(_mm512_cvtepi16_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::AVX512> cast_trunc(VecI16<Arch::AVX512> v) {
        __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
        return VecI64<Arch::AVX512>(_mm512_cvtepi16_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::AVX512> cast_trunc(VecI16<Arch::AVX512> v) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return VecF32<Arch::AVX512>(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(chunk)));
    }

    template<size_t N>
    inline VecF64<Arch::AVX512> cast_trunc(VecI16<Arch::AVX512> v) {
        __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
        __m512i i64 = _mm512_cvtepi16_epi64(chunk);
        return VecF64<Arch::AVX512>(_mm512_cvtepi64_pd(i64));
    }

    template<size_t N>
    inline VecI64<Arch::AVX512> cast_trunc(VecI32<Arch::AVX512> v) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return VecI64<Arch::AVX512>(_mm512_cvtepi32_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::AVX512> cast_trunc(VecI32<Arch::AVX512> v) {
        return VecF32<Arch::AVX512>(_mm512_cvtepi32_ps(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::AVX512> cast_trunc(VecI32<Arch::AVX512> v) {
        __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
        return VecF64<Arch::AVX512>(_mm512_cvtepi32_pd(chunk));
    }

    template<size_t N>
    inline VecF64<Arch::AVX512> cast_trunc(VecF32<Arch::AVX512> v) {
        __m256 chunk = (N == 0) ? _mm512_castps512_ps256(v.data) : _mm512_extractf32x8_ps(v.data, 1);
        return VecF64<Arch::AVX512>(_mm512_cvtps_pd(chunk));
    }

    template<>
    inline VecI32<Arch::AVX512> cast_trunc(VecI64<Arch::AVX512> v) {
        __m256i narrow = _mm512_cvtepi64_epi32(v.data);
        return VecI32<Arch::AVX512>(_mm512_castsi256_si512(narrow));
    }

    template<>
    inline VecI16<Arch::AVX512> cast_trunc(VecI64<Arch::AVX512> v) {
        __m128i narrow = _mm512_cvtepi64_epi16(v.data);
        return VecI16<Arch::AVX512>(_mm512_castsi128_si512(narrow));
    }

    template<>
    inline VecI8<Arch::AVX512> cast_trunc(VecI64<Arch::AVX512> v) {
        __m128i narrow = _mm512_cvtepi64_epi8(v.data);
        return VecI8<Arch::AVX512>(_mm512_castsi128_si512(narrow));
    }

    template<>
    inline VecI16<Arch::AVX512> cast_trunc(VecI32<Arch::AVX512> v) {
        __m256i narrow = _mm512_cvtepi32_epi16(v.data);
        return VecI16<Arch::AVX512>(_mm512_castsi256_si512(narrow));
    }

    template<>
    inline VecI8<Arch::AVX512> cast_trunc(VecI32<Arch::AVX512> v) {
        __m128i narrow = _mm512_cvtepi32_epi8(v.data);
        return VecI8<Arch::AVX512>(_mm512_castsi128_si512(narrow));
    }

    template<>
    inline VecI8<Arch::AVX512> cast_trunc(VecI16<Arch::AVX512> v) {
        __m256i narrow = _mm512_cvtepi16_epi8(v.data);
        return VecI8<Arch::AVX512>(_mm512_castsi256_si512(narrow));
    }

    template<>
    inline VecF32<Arch::AVX512> cast_trunc(VecF64<Arch::AVX512> v) {
        __m256 narrow = _mm512_cvtpd_ps(v.data);
        return VecF32<Arch::AVX512>(_mm512_castps256_ps512(narrow));
    }

    template<>
    inline VecI32<Arch::AVX512> cast_trunc(VecF32<Arch::AVX512> v) {
        return VecI32<Arch::AVX512>(_mm512_cvttps_epi32(v.data));
    }

    template<>
    inline VecI64<Arch::AVX512> cast_trunc(VecF64<Arch::AVX512> v) {
        return VecI64<Arch::AVX512>(_mm512_cvttpd_epi64(v.data));
    }

    template<>
    inline VecI32<Arch::AVX512> cast_trunc(VecF64<Arch::AVX512> v) {
        __m256i narrow = _mm512_cvttpd_epi32(v.data);
        return VecI32<Arch::AVX512>(_mm512_castsi256_si512(narrow));
    }

    template<size_t N>
    inline VecI64<Arch::AVX512> cast_trunc(VecF32<Arch::AVX512> v) {
        __m256 chunk = (N == 0) ? _mm512_castps512_ps256(v.data) : _mm512_extractf32x8_ps(v.data, 1);
        return VecI64<Arch::AVX512>(_mm512_cvttps_epi64(chunk));
    }

    template<>
    inline VecI16<Arch::AVX512> cast_trunc(VecF32<Arch::AVX512> v) {
        __m512i i32 = _mm512_cvttps_epi32(v.data);
        __m256i narrow = _mm512_cvtepi32_epi16(i32);
        return VecI16<Arch::AVX512>(_mm512_castsi256_si512(narrow));
    }

    template<>
    inline VecI8<Arch::AVX512> cast_trunc(VecF32<Arch::AVX512> v) {
        __m512i i32 = _mm512_cvttps_epi32(v.data);
        __m128i narrow = _mm512_cvtepi32_epi8(i32);
        return VecI8<Arch::AVX512>(_mm512_castsi128_si512(narrow));
    }

    template<>
    inline VecI16<Arch::AVX512> cast_trunc(VecF64<Arch::AVX512> v) {
        __m256i i32 = _mm512_cvttpd_epi32(v.data);
        __m128i narrow = _mm256_cvtepi32_epi16(i32);
        return VecI16<Arch::AVX512>(_mm512_castsi128_si512(narrow));
    }

    template<>
    inline VecI8<Arch::AVX512> cast_trunc(VecF64<Arch::AVX512> v) {
        __m256i i32 = _mm512_cvttpd_epi32(v.data);
        __m128i narrow = _mm256_cvtepi32_epi8(i32);
        return VecI8<Arch::AVX512>(_mm512_castsi128_si512(narrow));
    }

#if HAS_FLOAT16
    template<size_t N>
    inline VecF32<Arch::AVX512> cast_trunc(VecF16<Arch::AVX512> v) {
        __m256h chunk = (N == 0) ? _mm512_castph512_ph256(v.data) : _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(v.data), 1));
        return VecF32<Arch::AVX512>(_mm512_cvtxph_ps(chunk));
    }

    template<size_t N>
    inline VecF64<Arch::AVX512> cast_trunc(VecF16<Arch::AVX512> v) {
        __m256h half = (N < 2) ? _mm512_castph512_ph256(v.data) : _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(v.data), 1));
        __m128h chunk = (N % 2 == 0) ? _mm256_castph256_ph128(half) : _mm_castpd_ph(_mm256_extractf64x2_pd(_mm256_castph_pd(half), 1));
        return VecF64<Arch::AVX512>(_mm512_cvtph_pd(chunk));
    }

    template<>
    inline VecF16<Arch::AVX512> cast_trunc(VecF32<Arch::AVX512> v) {
        __m256h narrow = _mm512_cvtxps_ph(v.data);
        return VecF16<Arch::AVX512>(_mm512_castph256_ph512(narrow));
    }

    template<>
    inline VecF16<Arch::AVX512> cast_trunc(VecF64<Arch::AVX512> v) {
        __m128h narrow = _mm512_cvtpd_ph(v.data);
        return VecF16<Arch::AVX512>(_mm512_castph128_ph512(narrow));
    }

    template<size_t N>
    inline VecI32<Arch::AVX512> cast_trunc(VecF16<Arch::AVX512> v) {
        __m256h chunk = (N == 0) ? _mm512_castph512_ph256(v.data) : _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(v.data), 1));
        return VecI32<Arch::AVX512>(_mm512_cvttph_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::AVX512> cast_trunc(VecF16<Arch::AVX512> v) {
        __m256h half = (N < 2) ? _mm512_castph512_ph256(v.data) : _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(v.data), 1));
        __m128h chunk = (N % 2 == 0) ? _mm256_castph256_ph128(half) : _mm_castpd_ph(_mm256_extractf64x2_pd(_mm256_castph_pd(half), 1));
        return VecI64<Arch::AVX512>(_mm512_cvttph_epi64(chunk));
    }

    template<>
    inline VecI16<Arch::AVX512> cast_trunc(VecF16<Arch::AVX512> v) {
        return VecI16<Arch::AVX512>(_mm512_cvttph_epi16(v.data));
    }

    template<>
    inline VecI8<Arch::AVX512> cast_trunc(VecF16<Arch::AVX512> v) {
        __m512i i16 = _mm512_cvttph_epi16(v.data);
        __m256i narrow = _mm512_cvtepi16_epi8(i16);
        return VecI8<Arch::AVX512>(_mm512_castsi256_si512(narrow));
    }
#endif

#elif defined(__AVX2__)
    template<size_t N>
    inline VecI16<Arch::AVX2> cast_trunc(VecI8<Arch::AVX2> v) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return VecI16<Arch::AVX2>(_mm256_cvtepi8_epi16(chunk));
    }

    template<size_t N>
    inline VecI32<Arch::AVX2> cast_trunc(VecI8<Arch::AVX2> v) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        return VecI32<Arch::AVX2>(_mm256_cvtepi8_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::AVX2> cast_trunc(VecI8<Arch::AVX2> v) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 4);
        __m128i chunk = _mm_srli_si128(lane, (N % 4) * 4);
        return VecI64<Arch::AVX2>(_mm256_cvtepi8_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::AVX2> cast_trunc(VecI8<Arch::AVX2> v) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        return VecF32<Arch::AVX2>(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(chunk)));
    }

    template<size_t N>
    inline VecF64<Arch::AVX2> cast_trunc(VecI8<Arch::AVX2> v) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 4);
        __m128i chunk = _mm_srli_si128(lane, (N % 4) * 4);
        __m256i i64 = _mm256_cvtepi8_epi64(chunk);
        return VecF64<Arch::AVX2>(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i64)));
    }

    template<size_t N>
    inline VecI32<Arch::AVX2> cast_trunc(VecI16<Arch::AVX2> v) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return VecI32<Arch::AVX2>(_mm256_cvtepi16_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::AVX2> cast_trunc(VecI16<Arch::AVX2> v) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        return VecI64<Arch::AVX2>(_mm256_cvtepi16_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::AVX2> cast_trunc(VecI16<Arch::AVX2> v) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return VecF32<Arch::AVX2>(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(chunk)));
    }

    template<size_t N>
    inline VecF64<Arch::AVX2> cast_trunc(VecI16<Arch::AVX2> v) {
        __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
        __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
        __m256i i64 = _mm256_cvtepi16_epi64(chunk);
        return VecF64<Arch::AVX2>(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i64)));
    }

    template<size_t N>
    inline VecI64<Arch::AVX2> cast_trunc(VecI32<Arch::AVX2> v) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return VecI64<Arch::AVX2>(_mm256_cvtepi32_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::AVX2> cast_trunc(VecI32<Arch::AVX2> v) {
        return VecF32<Arch::AVX2>(_mm256_cvtepi32_ps(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::AVX2> cast_trunc(VecI32<Arch::AVX2> v) {
        __m128i chunk = _mm256_extracti128_si256(v.data, N);
        return VecF64<Arch::AVX2>(_mm256_cvtepi32_pd(chunk));
    }

    template<size_t N>
    inline VecF64<Arch::AVX2> cast_trunc(VecF32<Arch::AVX2> v) {
        __m128 chunk = (N == 0) ? _mm256_castps256_ps128(v.data) : _mm256_extractf128_ps(v.data, 1);
        return VecF64<Arch::AVX2>(_mm256_cvtps_pd(chunk));
    }

    template<>
    inline VecI32<Arch::AVX2> cast_trunc(VecI64<Arch::AVX2> v) {
        __m256i shuffled = _mm256_shuffle_epi32(v.data, _MM_SHUFFLE(2, 0, 2, 0));
        return VecI32<Arch::AVX2>(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));
    }

    template<>
    inline VecI16<Arch::AVX2> cast_trunc(VecI64<Arch::AVX2> v) {
        __m128i lo = _mm256_castsi256_si128(v.data);
        __m128i hi = _mm256_extracti128_si256(v.data, 1);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 8, 1, 0
        );
        lo = _mm_shuffle_epi8(lo, shuf);
        hi = _mm_shuffle_epi8(hi, shuf);
        return VecI16<Arch::AVX2>(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));
    }

    template<>
    inline VecI8<Arch::AVX2> cast_trunc(VecI64<Arch::AVX2> v) {
        __m128i lo = _mm256_castsi256_si128(v.data);
        __m128i hi = _mm256_extracti128_si256(v.data, 1);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 0
        );
        lo = _mm_shuffle_epi8(lo, shuf);
        hi = _mm_shuffle_epi8(hi, shuf);
        return VecI8<Arch::AVX2>(_mm256_castsi128_si256(_mm_unpacklo_epi16(lo, hi)));
    }

    template<>
    inline VecI16<Arch::AVX2> cast_trunc(VecI32<Arch::AVX2> v) {
        __m256i shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        __m256i shuffled = _mm256_shuffle_epi8(v.data, shuf);
        return VecI16<Arch::AVX2>(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));
    }

    template<>
    inline VecI8<Arch::AVX2> cast_trunc(VecI32<Arch::AVX2> v) {
        __m128i lo = _mm256_castsi256_si128(v.data);
        __m128i hi = _mm256_extracti128_si256(v.data, 1);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        lo = _mm_shuffle_epi8(lo, shuf);
        hi = _mm_shuffle_epi8(hi, shuf);
        return VecI8<Arch::AVX2>(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));
    }

    template<>
    inline VecI8<Arch::AVX2> cast_trunc(VecI16<Arch::AVX2> v) {
        __m256i shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0,
            -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0
        );
        __m256i shuffled = _mm256_shuffle_epi8(v.data, shuf);
        return VecI8<Arch::AVX2>(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));
    }

    template<>
    inline VecF32<Arch::AVX2> cast_trunc(VecF64<Arch::AVX2> v) {
        __m128 narrow = _mm256_cvtpd_ps(v.data);
        return VecF32<Arch::AVX2>(_mm256_castps128_ps256(narrow));
    }

    template<>
    inline VecI32<Arch::AVX2> cast_trunc(VecF32<Arch::AVX2> v) {
        return VecI32<Arch::AVX2>(_mm256_cvttps_epi32(v.data));
    }

    template<>
    inline VecI32<Arch::AVX2> cast_trunc(VecF64<Arch::AVX2> v) {
        __m128i narrow = _mm256_cvttpd_epi32(v.data);
        return VecI32<Arch::AVX2>(_mm256_castsi128_si256(narrow));
    }

    template<>
    inline VecI16<Arch::AVX2> cast_trunc(VecF32<Arch::AVX2> v) {
        __m256i i32 = _mm256_cvttps_epi32(v.data);
        __m256i shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        __m256i shuffled = _mm256_shuffle_epi8(i32, shuf);
        return VecI16<Arch::AVX2>(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));
    }

    template<>
    inline VecI8<Arch::AVX2> cast_trunc(VecF32<Arch::AVX2> v) {
        __m256i i32 = _mm256_cvttps_epi32(v.data);
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        lo = _mm_shuffle_epi8(lo, shuf);
        hi = _mm_shuffle_epi8(hi, shuf);
        return VecI8<Arch::AVX2>(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));
    }

    template<>
    inline VecI16<Arch::AVX2> cast_trunc(VecF64<Arch::AVX2> v) {
        __m128i i32 = _mm256_cvttpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        return VecI16<Arch::AVX2>(_mm256_castsi128_si256(_mm_shuffle_epi8(i32, shuf)));
    }

    template<>
    inline VecI8<Arch::AVX2> cast_trunc(VecF64<Arch::AVX2> v) {
        __m128i i32 = _mm256_cvttpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        return VecI8<Arch::AVX2>(_mm256_castsi128_si256(_mm_shuffle_epi8(i32, shuf)));
    }

#elif defined(__SSE4_1__)
    template<size_t N>
    inline VecI16<Arch::SSE4_1> cast_trunc(VecI8<Arch::SSE4_1> v) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return VecI16<Arch::SSE4_1>(_mm_cvtepi8_epi16(chunk));
    }

    template<size_t N>
    inline VecI32<Arch::SSE4_1> cast_trunc(VecI8<Arch::SSE4_1> v) {
        __m128i chunk = _mm_srli_si128(v.data, N * 4);
        return VecI32<Arch::SSE4_1>(_mm_cvtepi8_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::SSE4_1> cast_trunc(VecI8<Arch::SSE4_1> v) {
        __m128i chunk = _mm_srli_si128(v.data, N * 2);
        return VecI64<Arch::SSE4_1>(_mm_cvtepi8_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::SSE4_1> cast_trunc(VecI8<Arch::SSE4_1> v) {
        __m128i chunk = _mm_srli_si128(v.data, N * 4);
        return VecF32<Arch::SSE4_1>(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(chunk)));
    }

    template<size_t N>
    inline VecF64<Arch::SSE4_1> cast_trunc(VecI8<Arch::SSE4_1> v) {
        __m128i chunk = _mm_srli_si128(v.data, N * 2);
        return VecF64<Arch::SSE4_1>(_mm_cvtepi32_pd(_mm_cvtepi8_epi32(chunk)));
    }

    template<size_t N>
    inline VecI32<Arch::SSE4_1> cast_trunc(VecI16<Arch::SSE4_1> v) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return VecI32<Arch::SSE4_1>(_mm_cvtepi16_epi32(chunk));
    }

    template<size_t N>
    inline VecI64<Arch::SSE4_1> cast_trunc(VecI16<Arch::SSE4_1> v) {
        __m128i chunk = _mm_srli_si128(v.data, N * 4);
        return VecI64<Arch::SSE4_1>(_mm_cvtepi16_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::SSE4_1> cast_trunc(VecI16<Arch::SSE4_1> v) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return VecF32<Arch::SSE4_1>(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(chunk)));
    }

    template<size_t N>
    inline VecF64<Arch::SSE4_1> cast_trunc(VecI16<Arch::SSE4_1> v) {
        __m128i chunk = _mm_srli_si128(v.data, N * 4);
        return VecF64<Arch::SSE4_1>(_mm_cvtepi32_pd(_mm_cvtepi16_epi32(chunk)));
    }

    template<size_t N>
    inline VecI64<Arch::SSE4_1> cast_trunc(VecI32<Arch::SSE4_1> v) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return VecI64<Arch::SSE4_1>(_mm_cvtepi32_epi64(chunk));
    }

    template<size_t N>
    inline VecF32<Arch::SSE4_1> cast_trunc(VecI32<Arch::SSE4_1> v) {
        return VecF32<Arch::SSE4_1>(_mm_cvtepi32_ps(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::SSE4_1> cast_trunc(VecI32<Arch::SSE4_1> v) {
        __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
        return VecF64<Arch::SSE4_1>(_mm_cvtepi32_pd(chunk));
    }

    template<size_t N>
    inline VecF64<Arch::SSE4_1> cast_trunc(VecF32<Arch::SSE4_1> v) {
        __m128 chunk = (N == 0) ? v.data : _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v.data), 8));
        return VecF64<Arch::SSE4_1>(_mm_cvtps_pd(chunk));
    }

    template<>
    inline VecI32<Arch::SSE4_1> cast_trunc(VecI64<Arch::SSE4_1> v) {
        __m128i shuffled = _mm_shuffle_epi32(v.data, _MM_SHUFFLE(3, 3, 2, 0));
        return VecI32<Arch::SSE4_1>(shuffled);
    }

    template<>
    inline VecI16<Arch::SSE4_1> cast_trunc(VecI64<Arch::SSE4_1> v) {
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 8, 1, 0
        );
        return VecI16<Arch::SSE4_1>(_mm_shuffle_epi8(v.data, shuf));
    }

    template<>
    inline VecI8<Arch::SSE4_1> cast_trunc(VecI64<Arch::SSE4_1> v) {
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 0
        );
        return VecI8<Arch::SSE4_1>(_mm_shuffle_epi8(v.data, shuf));
    }

    template<>
    inline VecI16<Arch::SSE4_1> cast_trunc(VecI32<Arch::SSE4_1> v) {
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        return VecI16<Arch::SSE4_1>(_mm_shuffle_epi8(v.data, shuf));
    }

    template<>
    inline VecI8<Arch::SSE4_1> cast_trunc(VecI32<Arch::SSE4_1> v) {
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        return VecI8<Arch::SSE4_1>(_mm_shuffle_epi8(v.data, shuf));
    }

    template<>
    inline VecI8<Arch::SSE4_1> cast_trunc(VecI16<Arch::SSE4_1> v) {
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0
        );
        return VecI8<Arch::SSE4_1>(_mm_shuffle_epi8(v.data, shuf));
    }

    template<>
    inline VecF32<Arch::SSE4_1> cast_trunc(VecF64<Arch::SSE4_1> v) {
        return VecF32<Arch::SSE4_1>(_mm_cvtpd_ps(v.data));
    }

    template<>
    inline VecI32<Arch::SSE4_1> cast_trunc(VecF32<Arch::SSE4_1> v) {
        return VecI32<Arch::SSE4_1>(_mm_cvttps_epi32(v.data));
    }

    template<>
    inline VecI32<Arch::SSE4_1> cast_trunc(VecF64<Arch::SSE4_1> v) {
        return VecI32<Arch::SSE4_1>(_mm_cvttpd_epi32(v.data));
    }

    template<>
    inline VecI16<Arch::SSE4_1> cast_trunc(VecF32<Arch::SSE4_1> v) {
        __m128i i32 = _mm_cvttps_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
        );
        return VecI16<Arch::SSE4_1>(_mm_shuffle_epi8(i32, shuf));
    }

    template<>
    inline VecI8<Arch::SSE4_1> cast_trunc(VecF32<Arch::SSE4_1> v) {
        __m128i i32 = _mm_cvttps_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
        );
        return VecI8<Arch::SSE4_1>(_mm_shuffle_epi8(i32, shuf));
    }

    template<>
    inline VecI16<Arch::SSE4_1> cast_trunc(VecF64<Arch::SSE4_1> v) {
        __m128i i32 = _mm_cvttpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 4, 1, 0
        );
        return VecI16<Arch::SSE4_1>(_mm_shuffle_epi8(i32, shuf));
    }

    template<>
    inline VecI8<Arch::SSE4_1> cast_trunc(VecF64<Arch::SSE4_1> v) {
        __m128i i32 = _mm_cvttpd_epi32(v.data);
        __m128i shuf = _mm_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0
        );
        return VecI8<Arch::SSE4_1>(_mm_shuffle_epi8(i32, shuf));
    }

#else
    template<size_t N>
    inline VecI16<Arch::GENERIC> cast_trunc(VecI8<Arch::GENERIC> v) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>(v.data));
    }

    template<size_t N>
    inline VecI32<Arch::GENERIC> cast_trunc(VecI8<Arch::GENERIC> v) {
        return VecI32<Arch::GENERIC>(static_cast<int32_t>(v.data));
    }

    template<size_t N>
    inline VecI64<Arch::GENERIC> cast_trunc(VecI8<Arch::GENERIC> v) {
        return VecI64<Arch::GENERIC>(static_cast<int64_t>(v.data));
    }

    template<size_t N>
    inline VecF32<Arch::GENERIC> cast_trunc(VecI8<Arch::GENERIC> v) {
        return VecF32<Arch::GENERIC>(static_cast<float>(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::GENERIC> cast_trunc(VecI8<Arch::GENERIC> v) {
        return VecF64<Arch::GENERIC>(static_cast<double>(v.data));
    }

    template<size_t N>
    inline VecI32<Arch::GENERIC> cast_trunc(VecI16<Arch::GENERIC> v) {
        return VecI32<Arch::GENERIC>(static_cast<int32_t>(v.data));
    }

    template<size_t N>
    inline VecI64<Arch::GENERIC> cast_trunc(VecI16<Arch::GENERIC> v) {
        return VecI64<Arch::GENERIC>(static_cast<int64_t>(v.data));
    }

    template<size_t N>
    inline VecF32<Arch::GENERIC> cast_trunc(VecI16<Arch::GENERIC> v) {
        return VecF32<Arch::GENERIC>(static_cast<float>(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::GENERIC> cast_trunc(VecI16<Arch::GENERIC> v) {
        return VecF64<Arch::GENERIC>(static_cast<double>(v.data));
    }

    template<size_t N>
    inline VecI64<Arch::GENERIC> cast_trunc(VecI32<Arch::GENERIC> v) {
        return VecI64<Arch::GENERIC>(static_cast<int64_t>(v.data));
    }

    template<size_t N>
    inline VecF32<Arch::GENERIC> cast_trunc(VecI32<Arch::GENERIC> v) {
        return VecF32<Arch::GENERIC>(static_cast<float>(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::GENERIC> cast_trunc(VecI32<Arch::GENERIC> v) {
        return VecF64<Arch::GENERIC>(static_cast<double>(v.data));
    }

    template<size_t N>
    inline VecF64<Arch::GENERIC> cast_trunc(VecF32<Arch::GENERIC> v) {
        return VecF64<Arch::GENERIC>(static_cast<double>(v.data));
    }

    template<>
    inline VecI32<Arch::GENERIC> cast_trunc(VecI64<Arch::GENERIC> v) {
        return VecI32<Arch::GENERIC>(static_cast<int32_t>(v.data));
    }

    template<>
    inline VecI16<Arch::GENERIC> cast_trunc(VecI64<Arch::GENERIC> v) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>(v.data));
    }

    template<>
    inline VecI8<Arch::GENERIC> cast_trunc(VecI64<Arch::GENERIC> v) {
        return VecI8<Arch::GENERIC>(static_cast<int8_t>(v.data));
    }

    template<>
    inline VecI16<Arch::GENERIC> cast_trunc(VecI32<Arch::GENERIC> v) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>(v.data));
    }

    template<>
    inline VecI8<Arch::GENERIC> cast_trunc(VecI32<Arch::GENERIC> v) {
        return VecI8<Arch::GENERIC>(static_cast<int8_t>(v.data));
    }

    template<>
    inline VecI8<Arch::GENERIC> cast_trunc(VecI16<Arch::GENERIC> v) {
        return VecI8<Arch::GENERIC>(static_cast<int8_t>(v.data));
    }

    template<>
    inline VecF32<Arch::GENERIC> cast_trunc(VecF64<Arch::GENERIC> v) {
        return VecF32<Arch::GENERIC>(static_cast<float>(v.data));
    }

    template<>
    inline VecI32<Arch::GENERIC> cast_trunc(VecF32<Arch::GENERIC> v) {
        return VecI32<Arch::GENERIC>(static_cast<int32_t>(v.data));
    }

    template<>
    inline VecI64<Arch::GENERIC> cast_trunc(VecF64<Arch::GENERIC> v) {
        return VecI64<Arch::GENERIC>(static_cast<int64_t>(v.data));
    }

    template<>
    inline VecI32<Arch::GENERIC> cast_trunc(VecF64<Arch::GENERIC> v) {
        return VecI32<Arch::GENERIC>(static_cast<int32_t>(v.data));
    }

    template<size_t N>
    inline VecI64<Arch::GENERIC> cast_trunc(VecF32<Arch::GENERIC> v) {
        return VecI64<Arch::GENERIC>(static_cast<int64_t>(v.data));
    }

    template<>
    inline VecI16<Arch::GENERIC> cast_trunc(VecF32<Arch::GENERIC> v) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>(v.data));
    }

    template<>
    inline VecI8<Arch::GENERIC> cast_trunc(VecF32<Arch::GENERIC> v) {
        return VecI8<Arch::GENERIC>(static_cast<int8_t>(v.data));
    }

    template<>
    inline VecI16<Arch::GENERIC> cast_trunc(VecF64<Arch::GENERIC> v) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>(v.data));
    }

    template<>
    inline VecI8<Arch::GENERIC> cast_trunc(VecF64<Arch::GENERIC> v) {
        return VecI8<Arch::GENERIC>(static_cast<int8_t>(v.data));
    }
#endif

    template<typename T, size_t N = 0, typename F>
        requires ScalarCastable<T>
    inline vec<T> cast_trunc(F v) {
        return cast_trunc<vec<T>, N>(v);
    }
}

