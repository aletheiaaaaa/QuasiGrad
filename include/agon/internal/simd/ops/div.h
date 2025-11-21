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
    // Division (only available for floating-point types)
    template<typename Vec>
    inline Vec div(const Vec& a, const Vec& b);

#if defined(__AVX512F__)
#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> div(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_div_ph(a.data, b.data));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> div(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_div_ps(a.data, b.data));
    }

    template<>
    inline VecF64<Arch::AVX512> div(const VecF64<Arch::AVX512>& a, const VecF64<Arch::AVX512>& b) {
        return VecF64<Arch::AVX512>(_mm512_div_pd(a.data, b.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecF32<Arch::AVX2> div(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_div_ps(a.data, b.data));
    }

    template<>
    inline VecF64<Arch::AVX2> div(const VecF64<Arch::AVX2>& a, const VecF64<Arch::AVX2>& b) {
        return VecF64<Arch::AVX2>(_mm256_div_pd(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecF32<Arch::SSE4_1> div(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_div_ps(a.data, b.data));
    }

    template<>
    inline VecF64<Arch::SSE4_1> div(const VecF64<Arch::SSE4_1>& a, const VecF64<Arch::SSE4_1>& b) {
        return VecF64<Arch::SSE4_1>(_mm_div_pd(a.data, b.data));
    }
#else
    template<>
    inline VecF32<Arch::GENERIC> div(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data / b.data);
    }

    template<>
    inline VecF64<Arch::GENERIC> div(const VecF64<Arch::GENERIC>& a, const VecF64<Arch::GENERIC>& b) {
        return VecF64<Arch::GENERIC>(a.data / b.data);
    }
#endif
}
