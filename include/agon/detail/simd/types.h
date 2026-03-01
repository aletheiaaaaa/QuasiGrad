#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "arch.h"
#if defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__SSE4_1__)
  #include <emmintrin.h>
#endif

namespace agon::simd {
  template<Arch arch, typename T>
  struct Vec;

#if defined(__AVX512F__)
  template<>
  struct Vec<Arch::AVX512, int8_t> {
    using scalar_type = int8_t;
    using is_vec = std::true_type;
    static constexpr size_t size = 64;
    __m512i data;

    Vec() = default;
    explicit Vec(__m512i val) : data(val) {}
  };

  template<>
  struct Vec<Arch::AVX512, int16_t> {
    using scalar_type = int16_t;
    static constexpr size_t size = 32;
    __m512i data;

    Vec() = default;
    explicit Vec(__m512i val) : data(val) {}
  };

  template<>
  struct Vec<Arch::AVX512, float> {
    using scalar_type = float;
    static constexpr size_t size = 16;
    __m512 data;

    Vec() = default;
    explicit Vec(__m512 val) : data(val) {}
  };

  template<>
  struct Vec<Arch::AVX512, double> {
    using scalar_type = double;
    static constexpr size_t size = 8;
    __m512d data;

    Vec() = default;
    explicit Vec(__m512d val) : data(val) {}
  };
#elif defined(__AVX2__)
  template<>
  struct Vec<Arch::AVX2, int8_t> {
    using scalar_type = int8_t;
    static constexpr size_t size = 32;
    __m256i data;

    Vec() = default;
    explicit Vec(__m256i val) : data(val) {}
  };

  template<>
  struct Vec<Arch::AVX2, int16_t> {
    using scalar_type = int16_t;
    static constexpr size_t size = 16;
    __m256i data;

    Vec() = default;
    explicit Vec(__m256i val) : data(val) {}
  };

  template<>
  struct Vec<Arch::AVX2, float> {
    using scalar_type = float;
    static constexpr size_t size = 8;
    __m256 data;

    Vec() = default;
    explicit Vec(__m256 val) : data(val) {}
  };

  template<>
  struct Vec<Arch::AVX2, double> {
    using scalar_type = double;
    static constexpr size_t size = 4;
    __m256d data;

    Vec() = default;
    explicit Vec(__m256d val) : data(val) {}
  };
#elif defined(__SSE4_1__)
  template<>
  struct Vec<Arch::SSE4_1, int8_t> {
    using scalar_type = int8_t;
    static constexpr size_t size = 16;
    __m128i data;

    Vec() = default;
    explicit Vec(__m128i val) : data(val) {}
  };

  template<>
  struct Vec<Arch::SSE4_1, int16_t> {
    using scalar_type = int16_t;
    static constexpr size_t size = 8;
    __m128i data;

    Vec() = default;
    explicit Vec(__m128i val) : data(val) {}
  };

  template<>
  struct Vec<Arch::SSE4_1, float> {
    using scalar_type = float;
    static constexpr size_t size = 4;
    __m128 data;

    Vec() = default;
    explicit Vec(__m128 val) : data(val) {}
  };

  template<>
  struct Vec<Arch::SSE4_1, double> {
    using scalar_type = double;
    static constexpr size_t size = 2;
    __m128d data;

    Vec() = default;
    explicit Vec(__m128d val) : data(val) {}
  };
#else
  template<>
  struct Vec<Arch::GENERIC, int8_t> {
    using scalar_type = int8_t;
    static constexpr size_t size = 1;
    int8_t data;

    Vec() = default;
    explicit Vec(int8_t val) : data(val) {}
  };

  template<>
  struct Vec<Arch::GENERIC, int16_t> {
    using scalar_type = int16_t;
    static constexpr size_t size = 1;
    int16_t data;

    Vec() = default;
    explicit Vec(int16_t val) : data(val) {}
  };

  template<>
  struct Vec<Arch::GENERIC, float> {
    using scalar_type = float;
    static constexpr size_t size = 1;
    float data;

    Vec() = default;
    explicit Vec(float val) : data(val) {}
  };

  template<>
  struct Vec<Arch::GENERIC, double> {
    using scalar_type = double;
    static constexpr size_t size = 1;
    double data;

    Vec() = default;
    explicit Vec(double val) : data(val) {}
  };
#endif

  template<typename T>
  concept IsOperable = std::is_same_v<T, int8_t>
    || std::is_same_v<T, int16_t> || std::is_same_v<T, float>
    || std::is_same_v<T, double>;
}
