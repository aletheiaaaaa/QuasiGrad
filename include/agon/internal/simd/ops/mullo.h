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
    // Low part of integer multiplication (available for I16, I32, and I64 on AVX512)
    template<typename Vec>
    inline Vec mullo(const Vec& a, const Vec& b);

#if defined(__AVX512F__)
    template<>
    inline VecI16<Arch::AVX512> mullo(const VecI16<Arch::AVX512>& a, const VecI16<Arch::AVX512>& b) {
        return VecI16<Arch::AVX512>(_mm512_mullo_epi16(a.data, b.data));
    }

    template<>
    inline VecI32<Arch::AVX512> mullo(const VecI32<Arch::AVX512>& a, const VecI32<Arch::AVX512>& b) {
        return VecI32<Arch::AVX512>(_mm512_mullo_epi32(a.data, b.data));
    }

    template<>
    inline VecI64<Arch::AVX512> mullo(const VecI64<Arch::AVX512>& a, const VecI64<Arch::AVX512>& b) {
        return VecI64<Arch::AVX512>(_mm512_mullo_epi64(a.data, b.data));
    }
#elif defined(__AVX2__)
    template<>
    inline VecI16<Arch::AVX2> mullo(const VecI16<Arch::AVX2>& a, const VecI16<Arch::AVX2>& b) {
        return VecI16<Arch::AVX2>(_mm256_mullo_epi16(a.data, b.data));
    }

    template<>
    inline VecI32<Arch::AVX2> mullo(const VecI32<Arch::AVX2>& a, const VecI32<Arch::AVX2>& b) {
        return VecI32<Arch::AVX2>(_mm256_mullo_epi32(a.data, b.data));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecI16<Arch::SSE4_1> mullo(const VecI16<Arch::SSE4_1>& a, const VecI16<Arch::SSE4_1>& b) {
        return VecI16<Arch::SSE4_1>(_mm_mullo_epi16(a.data, b.data));
    }

    template<>
    inline VecI32<Arch::SSE4_1> mullo(const VecI32<Arch::SSE4_1>& a, const VecI32<Arch::SSE4_1>& b) {
        return VecI32<Arch::SSE4_1>(_mm_mullo_epi32(a.data, b.data));
    }
#else
    template<>
    inline VecI16<Arch::GENERIC> mullo(const VecI16<Arch::GENERIC>& a, const VecI16<Arch::GENERIC>& b) {
        return VecI16<Arch::GENERIC>(static_cast<int16_t>(a.data * b.data));
    }

    template<>
    inline VecI32<Arch::GENERIC> mullo(const VecI32<Arch::GENERIC>& a, const VecI32<Arch::GENERIC>& b) {
        return VecI32<Arch::GENERIC>(a.data * b.data);
    }

    template<>
    inline VecI64<Arch::GENERIC> mullo(const VecI64<Arch::GENERIC>& a, const VecI64<Arch::GENERIC>& b) {
        return VecI64<Arch::GENERIC>(a.data * b.data);
    }
#endif
}
