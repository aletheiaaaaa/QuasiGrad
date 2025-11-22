#include "../include/agon/optimizer.h"

namespace agon::optim {
    template<typename... Params>
    Optimizer::Optimizer(Params&... params) : parameters{ &params... }, state{} {}

    Optimizer::Optimizer(std::initializer_list<IParameter*> params) : parameters(params), state{} {}

    void Optimizer::zero_grad() {
        for (auto& param : parameters) {
            param->zero_grad();
        }
    }

    void Optimizer::add_parameter(IParameter& param) {
        parameters.push_back(&param);
    }
}