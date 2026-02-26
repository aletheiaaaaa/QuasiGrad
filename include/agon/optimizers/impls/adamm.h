#pragma once

#include "../optimizer.h"
#include "../../detail/simd/ops.h"
#include "../../detail/simd/utils.h"

namespace agon::optim {
    struct AdaMMParams {
        float lr = 1e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.4f;
        float epsilon = 1e-8f;

        bool maximize = false;
        bool use_adazo = false;
    };

    template<typename DedupedTuple>
    struct AdaMMState : public OptimizerState {
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> momentum{};
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> velocity{};
    };

    template<typename... Ts>
    class AdaMM : public Optimizer<Ts...> {
        public:
            explicit AdaMM(ParameterPack<Ts...> parameters, AdaMMParams options = {})
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... param_vecs) {
                        (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
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

            void load_from_bin(const std::string& path_str);
            void save_to_bin(const std::string& path_str) const;

        private:
            AdaMMParams options_;
            AdaMMState<Ts...> state_;
    };
}