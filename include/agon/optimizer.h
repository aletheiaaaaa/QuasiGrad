#pragma once

#include "parameter.h"
#include "detail/simd/arch.h"

#include <variant>
#include <vector>
#include <cstdint>
#include <stdfloat>

namespace agon::optim {
    struct GradData {
#if HAS_FLOAT16
        std::variant<std::vector<std::float16_t>, std::vector<float>, std::vector<double>> data{};
#else
        std::variant<std::vector<float>, std::vector<double>> data{};
#endif
    };

    struct OptimizerState {
        int64_t step = 0;
    };

    class Optimizer {
        public:
            template<typename... Params>
            explicit Optimizer(Params&... params);
            explicit Optimizer(std::initializer_list<IParameter*> params);

            void zero_grad();
            void add_parameter(IParameter& param);

            virtual void step() = 0;

            virtual void load_from_bin(const std::string& path) = 0;
            virtual void save_to_bin(const std::string& path) const = 0;

            ~Optimizer() = default;
        protected:
            OptimizerState state_;
            std::vector<IParameter*> parameters_;
    };
}
