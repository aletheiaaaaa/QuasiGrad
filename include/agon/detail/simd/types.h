#pragma once

#include <cstdint>
#include <cstddef>
#include <stdfloat>

#include "arch.h"

#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE4_1__)
    #include <emmintrin.h>
#endif

namespace agon::simd {
    template<Arch arch>
    struct VecI8;

    template<Arch arch>
    struct VecI16;

    template<Arch arch>
    struct VecI32;

    template<Arch arch>
    struct VecI64;

    template<Arch arch>
    struct VecF16;

    template<Arch arch>
    struct VecF32;

    template<Arch arch>
    struct VecF64;

#if defined(__AVX512F__)
    template<>
    struct VecI8<Arch::AVX512> {
        using scalar_type = int8_t;
        static constexpr size_t size = 64;
        __m512i data;

        VecI8() = default;
        explicit VecI8(__m512i val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::AVX512> {
        using scalar_type = int16_t;
        static constexpr size_t size = 32;
        __m512i data;

        VecI16() = default;
        explicit VecI16(__m512i val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::AVX512> {
        using scalar_type = int32_t;
        static constexpr size_t size = 16;
        __m512i data;

        VecI32() = default;
        explicit VecI32(__m512i val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::AVX512> {
        using scalar_type = int64_t;
        static constexpr size_t size = 8;
        __m512i data;

        VecI64() = default;
        explicit VecI64(__m512i val) : data(val) {}
    };

#if HAS_FLOAT16
    template<>
    struct VecF16<Arch::AVX512> {
        using scalar_type = std::float16_t;
        static constexpr size_t size = 32;
        __m512h data;

        VecF16() = default;
        explicit VecF16(__m512h val) : data(val) {}
    };
#endif

    template<>
    struct VecF32<Arch::AVX512> {
        using scalar_type = float;
        static constexpr size_t size = 16;
        __m512 data;

        VecF32() = default;
        explicit VecF32(__m512 val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::AVX512> {
        using scalar_type = double;
        static constexpr size_t size = 8;
        __m512d data;

        VecF64() = default;
        explicit VecF64(__m512d val) : data(val) {}
    };
#elif defined(__AVX2__)
    template<>
    struct VecI8<Arch::AVX2> {
        using scalar_type = int8_t;
        static constexpr size_t size = 32;
        __m256i data;

        VecI8() = default;
        explicit VecI8(__m256i val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::AVX2> {
        using scalar_type = int16_t;
        static constexpr size_t size = 16;
        __m256i data;

        VecI16() = default;
        explicit VecI16(__m256i val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::AVX2> {
        using scalar_type = int32_t;
        static constexpr size_t size = 8;
        __m256i data;

        VecI32() = default;
        explicit VecI32(__m256i val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::AVX2> {
        using scalar_type = int64_t;
        static constexpr size_t size = 4;
        __m256i data;

        VecI64() = default;
        explicit VecI64(__m256i val) : data(val) {}
    };

    template<>
    struct VecF32<Arch::AVX2> {
        using scalar_type = float;
        static constexpr size_t size = 8;
        __m256 data;

        VecF32() = default;
        explicit VecF32(__m256 val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::AVX2> {
        using scalar_type = double;
        static constexpr size_t size = 4;
        __m256d data;

        VecF64() = default;
        explicit VecF64(__m256d val) : data(val) {}
    };
#elif defined(__SSE4_1__)
    template<>
    struct VecI8<Arch::SSE4_1> {
        using scalar_type = int8_t;
        static constexpr size_t size = 16;
        __m128i data;

        VecI8() = default;
        explicit VecI8(__m128i val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::SSE4_1> {
        using scalar_type = int16_t;
        static constexpr size_t size = 8;
        __m128i data;

        VecI16() = default;
        explicit VecI16(__m128i val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::SSE4_1> {
        using scalar_type = int32_t;
        static constexpr size_t size = 4;
        __m128i data;

        VecI32() = default;
        explicit VecI32(__m128i val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::SSE4_1> {
        using scalar_type = int64_t;
        static constexpr size_t size = 2;
        __m128i data;

        VecI64() = default;
        explicit VecI64(__m128i val) : data(val) {}
    };

    template<>
    struct VecF32<Arch::SSE4_1> {
        using scalar_type = float;
        static constexpr size_t size = 4;
        __m128 data;

        VecF32() = default;
        explicit VecF32(__m128 val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::SSE4_1> {
        using scalar_type = double;
        static constexpr size_t size = 2;
        __m128d data;

        VecF64() = default;
        explicit VecF64(__m128d val) : data(val) {}
    };
#else
    template<>
    struct VecI8<Arch::GENERIC> {
        using scalar_type = int8_t;
        static constexpr size_t size = 1;
        int8_t data;

        VecI8() = default;
        explicit VecI8(int8_t val) : data(val) {}
    };

    template<>
    struct VecI16<Arch::GENERIC> {
        using scalar_type = int16_t;
        static constexpr size_t size = 1;
        int16_t data;

        VecI16() = default;
        explicit VecI16(int16_t val) : data(val) {}
    };

    template<>
    struct VecI32<Arch::GENERIC> {
        using scalar_type = int32_t;
        static constexpr size_t size = 1;
        int32_t data;

        VecI32() = default;
        explicit VecI32(int32_t val) : data(val) {}
    };

    template<>
    struct VecI64<Arch::GENERIC> {
        using scalar_type = int64_t;
        static constexpr size_t size = 1;
        int64_t data;

        VecI64() = default;
        explicit VecI64(int64_t val) : data(val) {}
    };

    template<>
    struct VecF32<Arch::GENERIC> {
        using scalar_type = float;
        static constexpr size_t size = 1;
        float data;

        VecF32() = default;
        explicit VecF32(float val) : data(val) {}
    };

    template<>
    struct VecF64<Arch::GENERIC> {
        using scalar_type = double;
        static constexpr size_t size = 1;
        double data;

        VecF64() = default;
        explicit VecF64(double val) : data(val) {}
    };
#endif

    template<typename T, Arch arch>
    struct VecType;

    template<Arch arch> struct VecType<int8_t, arch> { using type = VecI8<arch>; };
    template<Arch arch> struct VecType<int16_t, arch> { using type = VecI16<arch>; };
    template<Arch arch> struct VecType<int32_t, arch> { using type = VecI32<arch>; };
    template<Arch arch> struct VecType<int64_t, arch> { using type = VecI64<arch>; };
    template<Arch arch> struct VecType<float, arch> { using type = VecF32<arch>; };
    template<Arch arch> struct VecType<double, arch> { using type = VecF64<arch>; };

    template<Arch arch> struct VecType<std::float32_t, arch> { using type = VecF32<arch>; };
    template<Arch arch> struct VecType<std::float64_t, arch> { using type = VecF64<arch>; };
#if HAS_FLOAT16
    template<Arch arch> struct VecType<std::float16_t, arch> { using type = VecF16<arch>; };
#endif

    template<typename T>
    using vec = typename VecType<T, CURRENT_ARCH>::type;

    template<typename T>
concept ScalarCastable = std::is_same_v<T, int8_t>
    || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>
    || std::is_same_v<T, int64_t> || std::is_same_v<T, float>
    || std::is_same_v<T, double> || std::is_same_v<T, std::float32_t>
    || std::is_same_v<T, std::float64_t>
#if HAS_FLOAT16
    || std::is_same_v<T, std::float16_t>
#endif
    ;
}