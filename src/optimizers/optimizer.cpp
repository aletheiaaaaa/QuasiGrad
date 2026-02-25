#include "agon/optimizers/optimizer.h"

namespace agon::optim {
    template<typename... Ts>
    void Optimizer<Ts...>::zero_grad() {
        std::apply([](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [](auto& param_ref) {
                auto& param = param_ref.get();
                param.zero_grad();
            }), ...);
        }, parameters_.data);
    }

    template class Optimizer<std::tuple<agon::Parameter<float>>>;
    template class Optimizer<std::tuple<agon::Parameter<double>>>;
    template class Optimizer<std::tuple<agon::Parameter<float>, agon::Parameter<double>>>;
}