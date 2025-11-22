#include "../include/agon/estimators/rge.h"
#include "../include/agon/detail/simd/utils.h"
#include "../include/agon/detail/simd/ops.h"

namespace agon::estim {
    template<typename... Params>
    RGE::RGE(
        Params&... params,
        bool two_sided,
        int num_directions
    ) : options{two_sided, num_directions} {
        (parameters.push_back(&params), ...);
    }

    RGE::RGE(
        std::initializer_list<IParameter*> params,
        bool two_sided,
        int num_directions
    ) : options{two_sided, num_directions}, parameters(params) {}

    bool RGE::needs_eval() {
        return state.calls % (options.two_sided ? options.num_directions * 2 : options.num_directions) == 0;
    }

    void RGE::perturb() {
        // TODO
    }

    void RGE::observe(double value) {
        state.losses.push_back(value);
    }

    void RGE::finalize() {
        // TODO
    }
}