#include "../include/agon/estimator.h"
#include "../include/agon/detail/simd/utils.h"
#include "../include/agon/detail/simd/ops.h"

namespace agon::estim {
    template<typename... Params>
    Estimator::Estimator(Params&... params) : state_{}, parameters_{ &params... } {}

    Estimator::Estimator(std::initializer_list<IParameter*> params) : state_{}, parameters_(params) {}


    void Estimator::add_parameter(IParameter& param) {
        parameters_.push_back(&param);
    }

    void Estimator::clip_grad_norm(double max_norm) {
        if (max_norm <= 0.0) {
            return;
        }

        double total_norm_sq = 0.0;

            for (IParameter* param : parameters_) {
                simd::dispatch(param->dtype(), [&]<typename G>() {
                    G* grad_ptr = static_cast<G*>(param->grad_ptr());
                    size_t param_size = param->size();

                    constexpr size_t vec_size = simd::vec<G>::size;
                    constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                    size_t i = 0;
                    auto sum_vec = simd::setzero<simd::vec<G>>();

                    for (; i + vec_size * unroll_factor <= param_size; i += vec_size * unroll_factor) {
                        simd::unroll<unroll_factor>([&]<size_t index>() {
                            constexpr size_t offset = index * vec_size;

                            auto grad_vec = simd::load<simd::vec<G>>(&grad_ptr[i + offset]);
                            sum_vec = simd::fmadd(grad_vec, grad_vec, sum_vec);
                        });
                    }

                    G local_sum = simd::reduce_add(sum_vec);
                    for (; i < param_size; ++i) {
                        local_sum += grad_ptr[i] * grad_ptr[i];
                    }

                    total_norm_sq += static_cast<double>(local_sum);
                });
            }

        double total_norm = std::sqrt(total_norm_sq);
        if (total_norm > max_norm) {
            double scale = max_norm / (total_norm + 1e-6);

            for (IParameter* param : parameters_) {
                simd::dispatch(param->dtype(), [&]<typename G>() {
                    G* grad_ptr = static_cast<G*>(param->grad_ptr());
                    size_t param_size = param->size();

                    constexpr size_t vec_size = simd::vec<G>::size;
                    constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                    size_t i = 0;
                    auto scale_vec = simd::set1<simd::vec<G>>(static_cast<G>(scale));

                    for (; i + vec_size * unroll_factor <= param_size; i += vec_size * unroll_factor) {
                        simd::unroll<unroll_factor>([&]<size_t index>() {
                            constexpr size_t offset = index * vec_size;

                            auto grad_vec = simd::load<simd::vec<G>>(&grad_ptr[i + offset]);
                            grad_vec = simd::mul(grad_vec, scale_vec);
                            simd::store(&grad_ptr[i + offset], grad_vec);
                        });
                    }

                    for (; i < param_size; ++i) {
                        grad_ptr[i] = static_cast<G>(grad_ptr[i] * static_cast<G>(scale));
                    }
                });
            }
        }
    }
}