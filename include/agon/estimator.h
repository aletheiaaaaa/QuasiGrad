#pragma once

#include "parameter.h"

#include <vector>

namespace agon::estim {
    struct EstimatorState {
        size_t calls = 0;
    };

    class Estimator {
        public:
            template<typename... Params>
            explicit Estimator(Params&... params);
            explicit Estimator(std::initializer_list<IParameter*> params);

            void add_parameter(IParameter& param);
            void clip_grad_norm(double max_norm);

            virtual bool needs_eval() = 0;
            virtual void perturb() = 0;
            virtual void observe(double value) = 0;
            virtual void finalize() = 0;

            virtual ~Estimator() = default;
        protected:
            EstimatorState state_;
            std::vector<IParameter*> parameters_;
    };
}