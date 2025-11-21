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
    // High part of signed integer multiplication (only available for I16)
    template<typename Vec>
    inline Vec mulhi(const Vec& a, const Vec& b);

#if defined(__AVX512F__)
    template<>
    inline VecI16<Arch::AVX512> mulhi(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_mulhi_epi16(a.data, b.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecI16<Arch::AVX2> mulhi(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_mulhi_epi16(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecI16<Arch::SSE4_1> mulhi(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_mulhi_epi16(a.data, b.data));
    }
#else
    template<>
    inline VecI16<Arch::GENERIC> mulhi(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>((static_cast<int32_t>(a.data) * static_cast<int32_t>(b.data)) >> 16));
    }
#endif
}
