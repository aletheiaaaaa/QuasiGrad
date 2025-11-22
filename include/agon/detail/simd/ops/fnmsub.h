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
    template<typename Vec>
    inline Vec fnmsub(const Vec& a, const Vec& b, const Vec& c);

#if defined(__AVX512F__)
#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> fnmsub(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b, const VecF16<Arch::AVX512>& c) {
        return VecF16<Arch::AVX512>(_mm512_fnmsub_ph(a.data, b.data, c.data));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> fnmsub(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b, const VecF32<Arch::AVX512>& c) {
        return VecF32<Arch::AVX512>(_mm512_fnmsub_ps(a.data, b.data, c.data));
    }

    template<>
    inline VecF64<Arch::AVX512> fnmsub(const VecF64<Arch::AVX512>& a, const VecF64<Arch::AVX512>& b, const VecF64<Arch::AVX512>& c) {
        return VecF64<Arch::AVX512>(_mm512_fnmsub_pd(a.data, b.data, c.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecF32<Arch::AVX2> fnmsub(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b, const VecF32<Arch::AVX2>& c) {
        return VecF32<Arch::AVX2>(_mm256_fnmsub_ps(a.data, b.data, c.data));
    }

    template<>
    inline VecF64<Arch::AVX2> fnmsub(const VecF64<Arch::AVX2>& a, const VecF64<Arch::AVX2>& b, const VecF64<Arch::AVX2>& c) {
        return VecF64<Arch::AVX2>(_mm256_fnmsub_pd(a.data, b.data, c.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecF32<Arch::SSE4_1> fnmsub(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b, const VecF32<Arch::SSE4_1>& c) {
        __m128 neg_ab = _mm_sub_ps(_mm_setzero_ps(), _mm_mul_ps(a.data, b.data));
        return VecF32<Arch::SSE4_1>(_mm_sub_ps(neg_ab, c.data));
    }

    template<>
    inline VecF64<Arch::SSE4_1> fnmsub(const VecF64<Arch::SSE4_1>& a, const VecF64<Arch::SSE4_1>& b, const VecF64<Arch::SSE4_1>& c) {
        __m128d neg_ab = _mm_sub_pd(_mm_setzero_pd(), _mm_mul_pd(a.data, b.data));
        return VecF64<Arch::SSE4_1>(_mm_sub_pd(neg_ab, c.data));
    }
#else
    template<>
    inline VecF32<Arch::GENERIC> fnmsub(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b, const VecF32<Arch::GENERIC>& c) {
        return VecF32<Arch::GENERIC>(-(a.data * b.data) - c.data);
    }

    template<>
    inline VecF64<Arch::GENERIC> fnmsub(const VecF64<Arch::GENERIC>& a, const VecF64<Arch::GENERIC>& b, const VecF64<Arch::GENERIC>& c) {
        return VecF64<Arch::GENERIC>(-(a.data * b.data) - c.data);
    }
#endif
}
