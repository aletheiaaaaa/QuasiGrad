#pragma once

#include "../optimizer.h"  

#include <stdfloat>
#include <variant>

namespace agon::optim {
    struct SGDParams {
        float lr = 0.01f;
        float momentum = 0.0f;

        bool nesterov = false;
        bool maximize = false;
    };

    struct SGDState : public OptimizerState {
        std::vector<GradData> momenta;
    };

    class SGD : public Optimizer {
        public:
            template<class ...Params>
            explicit SGD(
                Params&... params, 
                float learning_rate = 0.01f, 
                float momentum = 0.0f, 
                bool nesterov = false, 
                bool maximize = false
            );
            explicit SGD(
                std::initializer_list<IParameter*> params, 
                float learning_rate = 0.01f, 
                float momentum = 0.0f, 
                bool nesterov = false, 
                bool maximize = false
            );

            void step() override;

            void load_from_bin(const std::string& path);
            void save_to_bin(const std::string& path) const;

        private:
            SGDParams options_;
            SGDState state_;
            std::vector<IParameter*> parameters_;
        };
}