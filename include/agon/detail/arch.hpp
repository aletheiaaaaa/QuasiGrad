#pragma once

#include <cstdint>

namespace agon::detail {
  enum class Arch : uint8_t {
    GENERIC,
    SSE4_1,
    AVX2,
    AVX512,
    NEON
  };

#if defined(__AVX512F__)
  inline constexpr Arch CURRENT_ARCH = Arch::AVX512;
#elif defined(__AVX2__)
  inline constexpr Arch CURRENT_ARCH = Arch::AVX2;
#elif defined(__SSE4_1__)
  inline constexpr Arch CURRENT_ARCH = Arch::SSE4_1;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  inline constexpr Arch CURRENT_ARCH = Arch::NEON;
#else
  inline constexpr Arch CURRENT_ARCH = Arch::GENERIC;
#endif
}