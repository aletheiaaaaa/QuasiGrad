#pragma once

#include "../estimator.h"

namespace agon::estim {
    struct RGEParams {
        bool two_sided = true;
        int num_directions = 1;
    };

    struct RGEState : public EstimatorState {
        std::vector<double> losses;

    };

    class RGE : public Estimator {
        public:
            template<typename... Params>
            explicit RGE(
                Params&... params,
                bool two_sided = true,
                int num_directions = 1
            );

            explicit RGE(
                std::initializer_list<IParameter*> params,
                bool two_sided = true,
                int num_directions = 1
            );

            bool needs_eval() override;
            void perturb() override;
            void observe(double value) override;
            void finalize() override;
        private:
            RGEParams options_;
            RGEState state_;
            std::vector<IParameter*> parameters_;
    };
}