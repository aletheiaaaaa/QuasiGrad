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
    inline void store(typename Vec::scalar_type* ptr, const Vec& v);

#if defined(__AVX512F__)
    template<>
    inline void store(int8_t* ptr, const VecI8<Arch::AVX512>& v) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
    }

    template<>
    inline void store(int16_t* ptr, const VecI16<Arch::AVX512>& v) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
    }

    template<>
    inline void store(int32_t* ptr, const VecI32<Arch::AVX512>& v) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
    }

    template<>
    inline void store(int64_t* ptr, const VecI64<Arch::AVX512>& v) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
    }

#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline void store(_Float16* ptr, const VecF16<Arch::AVX512>& v) {
        _mm512_storeu_ph(ptr, v.data);
    }
#endif

    template<>
    inline void store(float* ptr, const VecF32<Arch::AVX512>& v) {
        _mm512_storeu_ps(ptr, v.data);
    }

    template<>
    inline void store(double* ptr, const VecF64<Arch::AVX512>& v) {
        _mm512_storeu_pd(ptr, v.data);
    }
#elif defined(__AVX2__)
    template<>
    inline void store(int8_t* ptr, const VecI8<Arch::AVX2>& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
    }

    template<>
    inline void store(int16_t* ptr, const VecI16<Arch::AVX2>& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
    }

    template<>
    inline void store(int32_t* ptr, const VecI32<Arch::AVX2>& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
    }

    template<>
    inline void store(int64_t* ptr, const VecI64<Arch::AVX2>& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
    }

    template<>
    inline void store(float* ptr, const VecF32<Arch::AVX2>& v) {
        _mm256_storeu_ps(ptr, v.data);
    }

    template<>
    inline void store(double* ptr, const VecF64<Arch::AVX2>& v) {
        _mm256_storeu_pd(ptr, v.data);
    }
#elif defined(__SSE4_1__)
    template<>
    inline void store(int8_t* ptr, const VecI8<Arch::SSE4_1>& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
    }

    template<>
    inline void store(int16_t* ptr, const VecI16<Arch::SSE4_1>& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
    }

    template<>
    inline void store(int32_t* ptr, const VecI32<Arch::SSE4_1>& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
    }

    template<>
    inline void store(int64_t* ptr, const VecI64<Arch::SSE4_1>& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
    }

    template<>
    inline void store(float* ptr, const VecF32<Arch::SSE4_1>& v) {
        _mm_storeu_ps(ptr, v.data);
    }

    template<>
    inline void store(double* ptr, const VecF64<Arch::SSE4_1>& v) {
        _mm_storeu_pd(ptr, v.data);
    }
#else
    template<>
    inline void store(int8_t* ptr, const VecI8<Arch::GENERIC>& v) {
        *ptr = v.data;
    }

    template<>
    inline void store(int16_t* ptr, const VecI16<Arch::GENERIC>& v) {
        *ptr = v.data;
    }

    template<>
    inline void store(int32_t* ptr, const VecI32<Arch::GENERIC>& v) {
        *ptr = v.data;
    }

    template<>
    inline void store(int64_t* ptr, const VecI64<Arch::GENERIC>& v) {
        *ptr = v.data;
    }

    template<>
    inline void store(float* ptr, const VecF32<Arch::GENERIC>& v) {
        *ptr = v.data;
    }

    template<>
    inline void store(double* ptr, const VecF64<Arch::GENERIC>& v) {
        *ptr = v.data;
    }
#endif
}
