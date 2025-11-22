#pragma once

#include <cstddef>

#include "arch.h"

namespace agon::simd {
    constexpr size_t UNROLL_FACTOR = 
        (CURRENT_ARCH == Arch::AVX512) ? 4 :
        (CURRENT_ARCH == Arch::AVX2) ? 2 : 1;

    template<size_t N, typename F>
    constexpr void unroll(F&& func) {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (func.template operator()<Is>(), ...);
        }(std::make_index_sequence<N>{});
    }

    template<typename F>
    void dispatch_float(const std::type_info& type, F&& func) {
        if (type == typeid(std::float16_t)) {
            func.template operator()<std::float16_t>();
        } else if (type == typeid(std::float32_t)) {
            func.template operator()<std::float32_t>();
        } else if (type == typeid(std::float64_t)) {
            func.template operator()<std::float64_t>();
        } else {
            throw std::runtime_error("Unsupported data type for SIMD operation");
        }
    }
}