#include "../include/agon/optimizer.h"

#include <algorithm>

namespace agon::optim {
    template<typename... Ts>
    void Optimizer<Ts...>::zero_grad() {
        std::apply([](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [](auto& param) {
                param.zero_grad();
            }), ...);
        }, parameters_.data);
    }
}