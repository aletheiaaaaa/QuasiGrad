#pragma once

#include "../optimizer.h"

#include <algorithm>

namespace agon::optim {
    struct SpiderParams {
        bool maximize = false;
    };

    template<typename DedupedTuple>
    struct SpiderState : public OptimizerState {
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> prev_grad{};
    };

    template<typename... Ts>
    class Spider : public Optimizer<Ts...> {
        public:
            explicit Spider(ParameterPack<Ts...> parameters, SpiderParams options = {})
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... param_vecs) {
                        (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                            auto& param = param_ref.get();
                            using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                            auto& prev_grad = std::get<std::vector<T>>(this->state_.prev_grad);
                            prev_grad.insert(prev_grad.end(), param.size(), T(0));
                        }), ...);
                    }, this->parameters_.data);
                }

            void step() override;

            void load_from_bin(const std::string& path_str) override;
            void save_to_bin(const std::string& path_str) const override;

        private:
            SpiderParams options_;
            SpiderState<Ts...> state_;
    };
}