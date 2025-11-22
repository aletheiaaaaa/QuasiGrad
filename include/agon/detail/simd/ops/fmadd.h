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
    inline Vec fmadd(const Vec& a, const Vec& b, const Vec& c);

#if defined(__AVX512F__)
#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> fmadd(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b, const VecF16<Arch::AVX512>& c) {
        return VecF16<Arch::AVX512>(_mm512_fmadd_ph(a.data, b.data, c.data));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> fmadd(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b, const VecF32<Arch::AVX512>& c) {
        return VecF32<Arch::AVX512>(_mm512_fmadd_ps(a.data, b.data, c.data));
    }

    template<>
    inline VecF64<Arch::AVX512> fmadd(const VecF64<Arch::AVX512>& a, const VecF64<Arch::AVX512>& b, const VecF64<Arch::AVX512>& c) {
        return VecF64<Arch::AVX512>(_mm512_fmadd_pd(a.data, b.data, c.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecF32<Arch::AVX2> fmadd(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b, const VecF32<Arch::AVX2>& c) {
        return VecF32<Arch::AVX2>(_mm256_fmadd_ps(a.data, b.data, c.data));
    }

    template<>
    inline VecF64<Arch::AVX2> fmadd(const VecF64<Arch::AVX2>& a, const VecF64<Arch::AVX2>& b, const VecF64<Arch::AVX2>& c) {
        return VecF64<Arch::AVX2>(_mm256_fmadd_pd(a.data, b.data, c.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecF32<Arch::SSE4_1> fmadd(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b, const VecF32<Arch::SSE4_1>& c) {
        return VecF32<Arch::SSE4_1>(_mm_add_ps(_mm_mul_ps(a.data, b.data), c.data));
    }

    template<>
    inline VecF64<Arch::SSE4_1> fmadd(const VecF64<Arch::SSE4_1>& a, const VecF64<Arch::SSE4_1>& b, const VecF64<Arch::SSE4_1>& c) {
        return VecF64<Arch::SSE4_1>(_mm_add_pd(_mm_mul_pd(a.data, b.data), c.data));
    }
#else
    template<>
    inline VecF32<Arch::GENERIC> fmadd(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b, const VecF32<Arch::GENERIC>& c) {
        return VecF32<Arch::GENERIC>(a.data * b.data + c.data);
    }

    template<>
    inline VecF64<Arch::GENERIC> fmadd(const VecF64<Arch::GENERIC>& a, const VecF64<Arch::GENERIC>& b, const VecF64<Arch::GENERIC>& c) {
        return VecF64<Arch::GENERIC>(a.data * b.data + c.data);
    }
#endif
}
