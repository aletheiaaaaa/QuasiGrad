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
    inline Vec add(const Vec& a, const Vec& b);

#if defined(__AVX512F__)
    template<>
    inline VecI8<Arch::AVX512> add(const VecI8<Arch::AVX512>& a, const VecI8<Arch::AVX512>& b) {
        return VecI8<Arch::AVX512>(_mm512_add_epi8(a.data, b.data));
    }

    template<>
    inline VecI16<Arch::AVX512> add(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_add_epi16(a.data, b.data));
    }

    template<>
    inline VecI32<Arch::AVX512> add(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_add_epi32(a.data, b.data));
    }

    template<>
    inline VecI64<Arch::AVX512> add(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_add_epi64(a.data, b.data));
    }

#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> add(const VecF16<Arch::AVX512>& a, const VecF16<Arch::AVX512>& b) {
        return VecF16<Arch::AVX512>(_mm512_add_ph(a.data, b.data));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> add(const VecF32<Arch::AVX512>& a, const VecF32<Arch::AVX512>& b) {
        return VecF32<Arch::AVX512>(_mm512_add_ps(a.data, b.data));
    }

    template<>
    inline VecF64<Arch::AVX512> add(const VecF64<Arch::AVX512>& a, const VecF64<Arch::AVX512>& b) {
        return VecF64<Arch::AVX512>(_mm512_add_pd(a.data, b.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecI8<Arch::AVX2> add(const VecI8<Arch::AVX2>& a, const VecI8<Arch::AVX2>& b) {
        return VecI8<Arch::AVX2>(_mm256_add_epi8(a.data, b.data));
    }

    template<>
    inline VecI16<Arch::AVX2> add(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_add_epi16(a.data, b.data));
    }

    template<>
    inline VecI32<Arch::AVX2> add(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_add_epi32(a.data, b.data));
    }

    template<>
    inline VecI64<Arch::AVX2> add(const VecI64<Arch::AVX2>& a, const VecI64<Arch::AVX2>& b) {
        return VecI64<Arch::AVX2>(_mm256_add_epi64(a.data, b.data));
    }

    template<>
    inline VecF32<Arch::AVX2> add(const VecF32<Arch::AVX2>& a, const VecF32<Arch::AVX2>& b) {
        return VecF32<Arch::AVX2>(_mm256_add_ps(a.data, b.data));
    }

    template<>
    inline VecF64<Arch::AVX2> add(const VecF64<Arch::AVX2>& a, const VecF64<Arch::AVX2>& b) {
        return VecF64<Arch::AVX2>(_mm256_add_pd(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecI8<Arch::SSE4_1> add(const VecI8<Arch::SSE4_1>& a, const VecI8<Arch::SSE4_1>& b) {
        return VecI8<Arch::SSE4_1>(_mm_add_epi8(a.data, b.data));
    }

    template<>
    inline VecI16<Arch::SSE4_1> add(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_add_epi16(a.data, b.data));
    }

    template<>
    inline VecI32<Arch::SSE4_1> add(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_add_epi32(a.data, b.data));
    }

    template<>
    inline VecI64<Arch::SSE4_1> add(const VecI64<Arch::SSE4_1>& a, const VecI64<Arch::SSE4_1>& b) {
        return VecI64<Arch::SSE4_1>(_mm_add_epi64(a.data, b.data));
    }

    template<>
    inline VecF32<Arch::SSE4_1> add(const VecF32<Arch::SSE4_1>& a, const VecF32<Arch::SSE4_1>& b) {
        return VecF32<Arch::SSE4_1>(_mm_add_ps(a.data, b.data));
    }

    template<>
    inline VecF64<Arch::SSE4_1> add(const VecF64<Arch::SSE4_1>& a, const VecF64<Arch::SSE4_1>& b) {
        return VecF64<Arch::SSE4_1>(_mm_add_pd(a.data, b.data));
    }
#else
    template<>
    inline VecI8<Arch::GENERIC> add(const VecI8<Arch::GENERIC>& a, const VecI8<Arch::GENERIC>& b) {
        return VecI8<Arch::GENERIC>(a.data + b.data);
    }

    template<>
    inline VecI16<Arch::GENERIC> add(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(a.data + b.data);
    }

    template<>
    inline VecI32<Arch::GENERIC> add(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data + b.data);
    }

    template<>
    inline VecI64<Arch::GENERIC> add(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data + b.data);
    }

    template<>
    inline VecF32<Arch::GENERIC> add(const VecF32<Arch::GENERIC>& a, const VecF32<Arch::GENERIC>& b) {
        return VecF32<Arch::GENERIC>(a.data + b.data);
    }

    template<>
    inline VecF64<Arch::GENERIC> add(const VecF64<Arch::GENERIC>& a, const VecF64<Arch::GENERIC>& b) {
        return VecF64<Arch::GENERIC>(a.data + b.data);
    }
#endif
}