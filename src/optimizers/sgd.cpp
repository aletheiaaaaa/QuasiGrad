#include "../include/agon/optimizers/sgd.h"
#include "../include/agon/parameter.h"
#include "../include/agon/detail/simd/ops.h"
#include "../include/agon/detail/simd/utils.h"

namespace agon::optim {
    template<class ...Params>
    SGD::SGD(
        Params&... params, 
        float learning_rate, 
        float momentum, 
        bool nesterov, 
        bool maximize
    ) : options{learning_rate, momentum, nesterov, maximize} {
        (parameters.push_back(&params), ...);
        state.momenta.resize(parameters.size());
    }

    SGD::SGD(
        std::initializer_list<IParameter*> params, 
        float learning_rate, 
        float momentum, 
        bool nesterov, 
        bool maximize
    ) : options{learning_rate, momentum, nesterov, maximize}, parameters(params) {
        state.momenta.resize(parameters.size());
    }

    void SGD::step() {
        for (size_t idx = 0; idx < parameters.size(); ++idx) {
            IParameter& param = *parameters[idx];
            GradData& momentum = state.momenta[idx];

            simd::dispatch_float(param.grad_type(), [&]<typename G>() {
                G* grad_ptr = static_cast<G*>(param.grad_ptr());
                G* data_ptr = static_cast<G*>(param.data_ptr());

                size_t param_size = param.size();
                constexpr size_t vec_size = simd::vec<G>::size;

                auto& mom_data = std::get<std::vector<G>>(momentum.data);

                size_t i = 0;
                for (; i + vec_size * simd::UNROLL_FACTOR <= param_size; i += vec_size * simd::UNROLL_FACTOR) {
                    simd::unroll<simd::UNROLL_FACTOR>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad_vec = simd::load<simd::vec<G>>(&grad_ptr[i + offset]);
                        auto mom_vec = simd::load<simd::vec<G>>(&mom_data[i + offset]);

                        if (options.maximize) {
                            grad_vec = simd::neg(grad_vec);
                        }

                        auto mom_coeff = simd::set1<simd::vec<G>>(static_cast<G>(options.momentum));
                        mom_vec = simd::fmadd(mom_coeff, mom_vec, grad_vec);
                        simd::store(&mom_data[i + offset], mom_vec);

                        if (options.nesterov) {
                            mom_vec = simd::fmadd(mom_coeff, mom_vec, grad_vec);
                        }

                        auto lr_vec = simd::set1<simd::vec<G>>(static_cast<G>(options.lr));

                        auto data_vec = simd::load<simd::vec<G>>(&data_ptr[i + offset]);
                        data_vec = simd::fmsub(lr_vec, mom_vec, data_vec);
                        simd::store(&data_ptr[i + offset], data_vec);
                    });
                }

                for (; i < param_size; ++i) {
                    G grad_val = grad_ptr[i];
                    if (options.maximize) {
                        grad_val = -grad_val;
                    }

                    G& mom_val = mom_data[i];
                    mom_val = options.momentum * mom_val + grad_val;

                    G update_val = mom_val;
                    if (options.nesterov) {
                        update_val = options.momentum * mom_val + grad_val;
                    }

                    data_ptr[i] -= options.lr * update_val;
                }
            });
        }

        state.step += 1;
    }
}
