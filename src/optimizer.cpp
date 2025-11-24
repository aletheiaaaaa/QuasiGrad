#include "../include/agon/optimizer.h"

namespace agon::optim {
    template<typename... Params>
    Optimizer::Optimizer(Params&... params) : state_{}, parameters_{ &params... } {}

    Optimizer::Optimizer(std::initializer_list<IParameter*> params) : state_{}, parameters_(params) {}


    void Optimizer::zero_grad() {
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }

    void Optimizer::add_parameter(IParameter& param) {
        parameters_.push_back(&param);
    }
}