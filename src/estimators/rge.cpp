#include "../include/agon/estimators/rge.h"
#include "../include/agon/detail/simd/utils.h"
#include "../include/agon/detail/simd/ops.h"

namespace agon::estim {
    template<typename... Params>
    RGE::RGE(
        Params&... params,
        bool two_sided,
        int num_directions
    ) : options_{two_sided, num_directions}, parameters_{ &params... } {
        state_.losses.reserve( (two_sided ? num_directions * 2 : num_directions) );
    }

    RGE::RGE(
        std::initializer_list<IParameter*> params,
        bool two_sided,
        int num_directions
    ) : options_{two_sided, num_directions}, parameters_(params) {
        state_.losses.reserve( (two_sided ? num_directions * 2 : num_directions) );
    }

    bool RGE::needs_eval() {
        return state_.calls % (options_.two_sided ? options_.num_directions * 2 : options_.num_directions) == 0;
    }

    void RGE::perturb() {
        // TODO
    }

    void RGE::observe(double value) {
        state_.losses.push_back(value);
    }

    void RGE::finalize() {
        // TODO
    }
}