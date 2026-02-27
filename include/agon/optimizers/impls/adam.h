#pragma once

#include "../optimizer.h"
#include "../../detail/simd/ops.h"
#include "../../detail/simd/utils.h"

namespace agon::optim {
    struct AdamParams {
        float lr = 1e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.4f;
        float epsilon = 1e-8f;

        bool maximize = false;
        bool use_adazo = false;
    };

    template<typename DedupedTuple>
    struct AdamState : public OptimizerState {
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> momentum{};
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> velocity{};
    };

    template<typename... Ts>
    class Adam : public Optimizer<Ts...> {
        public:
            explicit Adam(ParameterPack<Ts...> parameters, AdamParams options = {})
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... param_vecs) {
                        (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                            auto& param = param_ref.get();
                            using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                            auto& mom = std::get<std::vector<T>>(this->state_.momentum);
                            auto& vel = std::get<std::vector<T>>(this->state_.velocity);
                            mom.insert(mom.end(), param.size(), T(0));
                            vel.insert(vel.end(), param.size(), T(0));
                        }), ...);
                    }, this->parameters_.data);
                }

            void step() override;

            void load_from_bin(const std::string& path_str) override;
            void save_to_bin(const std::string& path_str) const override;

        private:
            AdamParams options_;
            AdamState<Ts...> state_;
    };
}