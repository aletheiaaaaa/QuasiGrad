#pragma once

#include <cstdint>

#if defined(__AVX512_FP16__)
    #define HAS_FLOAT16 1
#else
    #define HAS_FLOAT16 0
#endif

namespace agon::simd {
    enum class Arch : uint8_t {
        GENERIC,
        SSE4_1,
        AVX2,
        AVX512,
    };

    inline constexpr Arch detect_arch() {
#if defined(__AVX512F__)
        return Arch::AVX512;
#elif defined(__AVX2__)
        return Arch::AVX2;
#elif defined(__SSE4_1__)
        return Arch::SSE4_1;
#else
        return Arch::GENERIC;
#endif
    }

    constexpr Arch CURRENT_ARCH = detect_arch();
}