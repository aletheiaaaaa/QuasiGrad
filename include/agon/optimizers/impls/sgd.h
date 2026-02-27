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
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> momenta{};
    };

    template<typename... Ts>
    class SGD : public Optimizer<Ts...> {
        public:
            explicit SGD(ParameterPack<Ts...> parameters, SGDParams options = {}) 
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... param_vecs) {
                        (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                            auto& param = param_ref.get();
                            using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                            auto& mom = std::get<std::vector<T>>(this->state_.momenta);
                            mom.insert(mom.end(), param.size(), T(0));
                        }), ...);
                    }, this->parameters_.data);
                }

            void step() override;

            void load_from_bin(const std::string& path_str) override;
            void save_to_bin(const std::string& path_str) const override;

        private:
            SGDParams options_;
            SGDState<Ts...> state_;
    };
}