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
    inline Vec load(const typename Vec::scalar_type* ptr);

#if defined(__AVX512F__)
    template<>
    inline VecI8<Arch::AVX512> load(const int8_t* ptr) {
        return VecI8<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

    template<>
    inline VecI16<Arch::AVX512> load(const int16_t* ptr) {
        return VecI16<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

    template<>
    inline VecI32<Arch::AVX512> load(const int32_t* ptr) {
        return VecI32<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

    template<>
    inline VecI64<Arch::AVX512> load(const int64_t* ptr) {
        return VecI64<Arch::AVX512>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
    }

#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> load(const _Float16* ptr) {
        return VecF16<Arch::AVX512>(_mm512_loadu_ph(ptr));
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> load(const float* ptr) {
        return VecF32<Arch::AVX512>(_mm512_loadu_ps(ptr));
    }

    template<>
    inline VecF64<Arch::AVX512> load(const double* ptr) {
        return VecF64<Arch::AVX512>(_mm512_loadu_pd(ptr));
    }
#elif defined(__AVX2__)
    template<>
    inline VecI8<Arch::AVX2> load(const int8_t* ptr) {
        return VecI8<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    template<>
    inline VecI16<Arch::AVX2> load(const int16_t* ptr) {
        return VecI16<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    template<>
    inline VecI32<Arch::AVX2> load(const int32_t* ptr) {
        return VecI32<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    template<>
    inline VecI64<Arch::AVX2> load(const int64_t* ptr) {
        return VecI64<Arch::AVX2>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
    }

    template<>
    inline VecF32<Arch::AVX2> load(const float* ptr) {
        return VecF32<Arch::AVX2>(_mm256_loadu_ps(ptr));
    }

    template<>
    inline VecF64<Arch::AVX2> load(const double* ptr) {
        return VecF64<Arch::AVX2>(_mm256_loadu_pd(ptr));
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecI8<Arch::SSE4_1> load(const int8_t* ptr) {
        return VecI8<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    template<>
    inline VecI16<Arch::SSE4_1> load(const int16_t* ptr) {
        return VecI16<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    template<>
    inline VecI32<Arch::SSE4_1> load(const int32_t* ptr) {
        return VecI32<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    template<>
    inline VecI64<Arch::SSE4_1> load(const int64_t* ptr) {
        return VecI64<Arch::SSE4_1>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }

    template<>
    inline VecF32<Arch::SSE4_1> load(const float* ptr) {
        return VecF32<Arch::SSE4_1>(_mm_loadu_ps(ptr));
    }

    template<>
    inline VecF64<Arch::SSE4_1> load(const double* ptr) {
        return VecF64<Arch::SSE4_1>(_mm_loadu_pd(ptr));
    }
#else
    template<>
    inline VecI8<Arch::GENERIC> load(const int8_t* ptr) {
        return VecI8<Arch::GENERIC>(*ptr);
    }

    template<>
    inline VecI16<Arch::GENERIC> load(const int16_t* ptr) {
        return VecI16<Arch::GENERIC>(*ptr);
    }

    template<>
    inline VecI32<Arch::GENERIC> load(const int32_t* ptr) {
        return VecI32<Arch::GENERIC>(*ptr);
    }

    template<>
    inline VecI64<Arch::GENERIC> load(const int64_t* ptr) {
        return VecI64<Arch::GENERIC>(*ptr);
    }

    template<>
    inline VecF32<Arch::GENERIC> load(const float* ptr) {
        return VecF32<Arch::GENERIC>(*ptr);
    }

    template<>
    inline VecF64<Arch::GENERIC> load(const double* ptr) {
        return VecF64<Arch::GENERIC>(*ptr);
    }
#endif
}
