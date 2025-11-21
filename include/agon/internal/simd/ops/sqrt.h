#pragma once

#include "../arch.h"
#include "../types.h"

#include <cmath>

#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
#endif

namespace agon::simd {
    // Square root (only available for floating-point types)
    template<typename Vec>
    inline Vec sqrt(const Vec& a);

#if defined(__AVX512F__)
#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> sqrt(const VecF16<Arch::AVX512>& a) {
        return VecF16<Arch::AVX512>(_mm512_sqrt_ph(a.data));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> sqrt(const VecF32<Arch::AVX512>& a) {
        return VecF32<Arch::AVX512>(_mm512_sqrt_ps(a.data));
    }

    template<>
    inline VecF64<Arch::AVX512> sqrt(const VecF64<Arch::AVX512>& a) {
        return VecF64<Arch::AVX512>(_mm512_sqrt_pd(a.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecF32<Arch::AVX2> sqrt(const VecF32<Arch::AVX2>& a) {
        return VecF32<Arch::AVX2>(_mm256_sqrt_ps(a.data));
    }

    template<>
    inline VecF64<Arch::AVX2> sqrt(const VecF64<Arch::AVX2>& a) {
        return VecF64<Arch::AVX2>(_mm256_sqrt_pd(a.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecF32<Arch::SSE4_1> sqrt(const VecF32<Arch::SSE4_1>& a) {
        return VecF32<Arch::SSE4_1>(_mm_sqrt_ps(a.data));
    }

    template<>
    inline VecF64<Arch::SSE4_1> sqrt(const VecF64<Arch::SSE4_1>& a) {
        return VecF64<Arch::SSE4_1>(_mm_sqrt_pd(a.data));
    }
#else
    template<>
    inline VecF32<Arch::GENERIC> sqrt(const VecF32<Arch::GENERIC>& a) {
        return VecF32<Arch::GENERIC>(std::sqrt(a.data));
    }

    template<>
    inline VecF64<Arch::GENERIC> sqrt(const VecF64<Arch::GENERIC>& a) {
        return VecF64<Arch::GENERIC>(std::sqrt(a.data));
    }
#endif
}
