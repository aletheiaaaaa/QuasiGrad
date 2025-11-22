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
    inline Vec neg(const Vec& a);

#if defined(__AVX512F__)
#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> neg(const VecF16<Arch::AVX512>& a) {
        const __m512i sign_mask = _mm512_set1_epi16(static_cast<short>(0x8000));
        return VecF16<Arch::AVX512>(_mm512_castsi512_ph(_mm512_xor_si512(_mm512_castph_si512(a.data), sign_mask)));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> neg(const VecF32<Arch::AVX512>& a) {
        return VecF32<Arch::AVX512>(_mm512_xor_ps(a.data, _mm512_set1_ps(-0.0f)));
    }

    template<>
    inline VecF64<Arch::AVX512> neg(const VecF64<Arch::AVX512>& a) {
        return VecF64<Arch::AVX512>(_mm512_xor_pd(a.data, _mm512_set1_pd(-0.0)));
    }
#elif defined(__AVX2__)
    template<>
    inline VecF32<Arch::AVX2> neg(const VecF32<Arch::AVX2>& a) {
        return VecF32<Arch::AVX2>(_mm256_xor_ps(a.data, _mm256_set1_ps(-0.0f)));
    }

    template<>
    inline VecF64<Arch::AVX2> neg(const VecF64<Arch::AVX2>& a) {
        return VecF64<Arch::AVX2>(_mm256_xor_pd(a.data, _mm256_set1_pd(-0.0)));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecF32<Arch::SSE4_1> neg(const VecF32<Arch::SSE4_1>& a) {
        return VecF32<Arch::SSE4_1>(_mm_xor_ps(a.data, _mm_set1_ps(-0.0f)));
    }

    template<>
    inline VecF64<Arch::SSE4_1> neg(const VecF64<Arch::SSE4_1>& a) {
        return VecF64<Arch::SSE4_1>(_mm_xor_pd(a.data, _mm_set1_pd(-0.0)));
    }
#else
    template<>
    inline VecF32<Arch::GENERIC> neg(const VecF32<Arch::GENERIC>& a) {
        return VecF32<Arch::GENERIC>(-a.data);
    }

    template<>
    inline VecF64<Arch::GENERIC> neg(const VecF64<Arch::GENERIC>& a) {
        return VecF64<Arch::GENERIC>(-a.data);
    }
#endif
}
