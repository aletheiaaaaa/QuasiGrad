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
    // Broadcast a single value to all elements
    template<typename Vec>
    inline Vec set1(typename Vec::scalar_type val);

#if defined(__AVX512F__)
    template<>
    inline VecI8<Arch::AVX512> set1(int8_t val) {
        return VecI8<Arch::AVX512>(_mm512_set1_epi8(val));
    }

    template<>
    inline VecI16<Arch::AVX512> set1(int16_t val) {
        return VecI16<Arch::AVX512>(_mm512_set1_epi16(val));
    }

    template<>
    inline VecI32<Arch::AVX512> set1(int32_t val) {
        return VecI32<Arch::AVX512>(_mm512_set1_epi32(val));
    }

    template<>
    inline VecI64<Arch::AVX512> set1(int64_t val) {
        return VecI64<Arch::AVX512>(_mm512_set1_epi64(val));
    }

#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> set1(_Float16 val) {
        return VecF16<Arch::AVX512>(_mm512_set1_ph(val));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> set1(float val) {
        return VecF32<Arch::AVX512>(_mm512_set1_ps(val));
    }

    template<>
    inline VecF64<Arch::AVX512> set1(double val) {
        return VecF64<Arch::AVX512>(_mm512_set1_pd(val));
    }
#elif defined(__AVX2__)
    template<>
    inline VecI8<Arch::AVX2> set1(int8_t val) {
        return VecI8<Arch::AVX2>(_mm256_set1_epi8(val));
    }

    template<>
    inline VecI16<Arch::AVX2> set1(int16_t val) {
        return VecI16<Arch::AVX2>(_mm256_set1_epi16(val));
    }

    template<>
    inline VecI32<Arch::AVX2> set1(int32_t val) {
        return VecI32<Arch::AVX2>(_mm256_set1_epi32(val));
    }

    template<>
    inline VecI64<Arch::AVX2> set1(int64_t val) {
        return VecI64<Arch::AVX2>(_mm256_set1_epi64x(val));
    }

    template<>
    inline VecF32<Arch::AVX2> set1(float val) {
        return VecF32<Arch::AVX2>(_mm256_set1_ps(val));
    }

    template<>
    inline VecF64<Arch::AVX2> set1(double val) {
        return VecF64<Arch::AVX2>(_mm256_set1_pd(val));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecI8<Arch::SSE4_1> set1(int8_t val) {
        return VecI8<Arch::SSE4_1>(_mm_set1_epi8(val));
    }

    template<>
    inline VecI16<Arch::SSE4_1> set1(int16_t val) {
        return VecI16<Arch::SSE4_1>(_mm_set1_epi16(val));
    }

    template<>
    inline VecI32<Arch::SSE4_1> set1(int32_t val) {
        return VecI32<Arch::SSE4_1>(_mm_set1_epi32(val));
    }

    template<>
    inline VecI64<Arch::SSE4_1> set1(int64_t val) {
        return VecI64<Arch::SSE4_1>(_mm_set1_epi64x(val));
    }

    template<>
    inline VecF32<Arch::SSE4_1> set1(float val) {
        return VecF32<Arch::SSE4_1>(_mm_set1_ps(val));
    }

    template<>
    inline VecF64<Arch::SSE4_1> set1(double val) {
        return VecF64<Arch::SSE4_1>(_mm_set1_pd(val));
    }
#else
    template<>
    inline VecI8<Arch::GENERIC> set1(int8_t val) {
        return VecI8<Arch::GENERIC>(val);
    }

    template<>
    inline VecI16<Arch::GENERIC> set1(int16_t val) {
        return VecI16<Arch::GENERIC>(val);
    }

    template<>
    inline VecI32<Arch::GENERIC> set1(int32_t val) {
        return VecI32<Arch::GENERIC>(val);
    }

    template<>
    inline VecI64<Arch::GENERIC> set1(int64_t val) {
        return VecI64<Arch::GENERIC>(val);
    }

    template<>
    inline VecF32<Arch::GENERIC> set1(float val) {
        return VecF32<Arch::GENERIC>(val);
    }

    template<>
    inline VecF64<Arch::GENERIC> set1(double val) {
        return VecF64<Arch::GENERIC>(val);
    }
#endif
}
