#pragma once

#include "parameter.h"

#include <vector>
#include <cstdint>

namespace agon::optim {
    struct GradData {
        std::variant<std::vector<std::float16_t>, std::vector<float>, std::vector<double>> data{};
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
            OptimizerState state;
            std::vector<IParameter*> parameters;
    };
}
