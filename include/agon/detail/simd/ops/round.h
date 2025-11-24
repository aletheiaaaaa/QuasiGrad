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
    template<typename Vec>
    inline Vec round(Vec v);

#if defined(__AVX512F__)
    template<>
    inline VecF32<Arch::AVX512> round(VecF32<Arch::AVX512> v) {
        return VecF32<Arch::AVX512>(_mm512_roundscale_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    template<>
    inline VecF64<Arch::AVX512> round(VecF64<Arch::AVX512> v) {
        return VecF64<Arch::AVX512>(_mm512_roundscale_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

#if HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> round(VecF16<Arch::AVX512> v) {
        return VecF16<Arch::AVX512>(_mm512_roundscale_ph(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
#endif

#elif defined(__AVX2__)
    template<>
    inline VecF32<Arch::AVX2> round(VecF32<Arch::AVX2> v) {
        return VecF32<Arch::AVX2>(_mm256_round_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    template<>
    inline VecF64<Arch::AVX2> round(VecF64<Arch::AVX2> v) {
        return VecF64<Arch::AVX2>(_mm256_round_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

#elif defined(__SSE4_1__)
    template<>
    inline VecF32<Arch::SSE4_1> round(VecF32<Arch::SSE4_1> v) {
        return VecF32<Arch::SSE4_1>(_mm_round_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    template<>
    inline VecF64<Arch::SSE4_1> round(VecF64<Arch::SSE4_1> v) {
        return VecF64<Arch::SSE4_1>(_mm_round_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

#else
    template<>
    inline VecF32<Arch::GENERIC> round(VecF32<Arch::GENERIC> v) {
        return VecF32<Arch::GENERIC>(std::round(v.data));
    }

    template<>
    inline VecF64<Arch::GENERIC> round(VecF64<Arch::GENERIC> v) {
        return VecF64<Arch::GENERIC>(std::round(v.data));
    }
#endif
}
