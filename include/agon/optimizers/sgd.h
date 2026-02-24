#pragma once

#include "../optimizer.h"  
#include <algorithm>

namespace agon::optim {
    struct SGDParams {
        float lr = 0.01f;
        float momentum = 0.0f;

        bool nesterov = false;
        bool maximize = false;
    };

    template<typename DedupedTuple>
    struct SGDState : public OptimizerState {
        dedup::TransformTuple_t<std::vector, DedupedTuple> momenta{};
    };

    template<typename... Ts>
    class SGD : public Optimizer<Ts...> {
        public:
            explicit SGD(ParameterPack<Ts...> parameters, SGDParams options = {}) 
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... params) {
                        (std::ranges::for_each(params.begin(), params.end(), [&](auto& param) {
                            using DataType = typename std::decay_t<decltype(param)>::DataType;
                            std::get<std::vector<DataType>>(this->state_.momenta).emplace_back(param.size());
                        }), ...);
                    }, this->parameters_.data);
                }

            void step() override;

            void load_from_bin(const std::string& path_str);
            void save_to_bin(const std::string& path_str) const;

        private:
            SGDParams options_;
            SGDState<Ts...> state_;
    };
}