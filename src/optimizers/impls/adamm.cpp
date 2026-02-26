#include "agon/optimizers/impls/adamm.h"

#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

#include <fstream>
#include <filesystem>

namespace agon::optim {
    template<typename... Ts>
    void AdaMM<Ts...>::step() {
        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

                auto& grad = param.grad();
                auto& data = param.data();
                auto& mom = std::get<std::vector<T>>(state_.momentum);
                auto& vel = std::get<std::vector<T>>(state_.velocity);

                constexpr size_t vec_size = simd::vec<T>::size;
                constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                size_t i = 0;
                for (; i + vec_size * unroll_factor <= grad.size(); i += vec_size * unroll_factor) {
                    simd::unroll<unroll_factor>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad_vec = simd::load<T>(&grad[i + offset]);
                        auto mom_vec = simd::load<T>(&mom[i + offset]);
                        auto vel_vec = simd::load<T>(&vel[i + offset]);

                        if (options_.maximize) grad_vec = simd::neg(grad_vec);

                        auto beta1_vec = simd::set1<T>(options_.beta1);
                        mom_vec = simd::fmadd(beta1_vec, mom_vec, grad_vec);
                        mom_vec = simd::fnmadd(beta1_vec, grad_vec, mom_vec);

                        auto beta2_vec = simd::set1<T>(options_.beta2);
                        auto grad_squared = (options_.use_adazo) ? simd::mul(mom_vec, mom_vec) : simd::mul(grad_vec, grad_vec);
                        vel_vec = simd::fmadd(beta2_vec, vel_vec, grad_squared);
                        vel_vec = simd::fnmadd(beta2_vec, grad_squared, vel_vec);

                        simd::store(&mom[i + offset], mom_vec);
                        simd::store(&vel[i + offset], vel_vec);

                        auto epsilon_vec = simd::set1<T>(options_.epsilon);
                        auto denom_vec = simd::add(simd::sqrt(vel_vec), epsilon_vec);
                        auto update_vec = simd::div(mom_vec, denom_vec);

                        auto lr_vec = simd::set1<T>(options_.lr);
                        auto data_vec = simd::load<T>(&data[i + offset]);
                        data_vec = simd::fmadd(lr_vec, update_vec, data_vec);
                        simd::store(&data[i + offset], data_vec);
                    });
                }

                for (; i < grad.size(); ++i) {
                    T grad_val = options_.maximize ? -grad[i] : grad[i];

                    T mom_val = options_.beta1 * mom[i] + (1 - options_.beta1) * grad_val;
                    T vel_val = options_.beta2 * vel[i] + (1 - options_.beta2) * grad_val * grad_val;

                    mom[i] = mom_val;
                    vel[i] = vel_val;

                    T update = mom_val / (std::sqrt(vel_val) + options_.epsilon);
                    data[i] += options_.lr * update;
                }
            }), ...);
        }, this->parameters_.data);

        state_.step++;
    }

    template<typename... Ts>
    void AdaMM<Ts...>::load_from_bin(const std::string& path_str) {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& mom = std::get<std::vector<T>>(state_.momentum);
                auto& vel = std::get<std::vector<T>>(state_.velocity);

                in.read(reinterpret_cast<char*>(&param.data()), param.size() * sizeof(T));
                in.read(reinterpret_cast<char*>(mom.data()), param.size() * sizeof(T));
                in.read(reinterpret_cast<char*>(vel.data()), param.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template<typename... Ts>
    void AdaMM<Ts...>::save_to_bin(const std::string& path_str) const {
        std::filesystem::path path(path_str);
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& mom = std::get<std::vector<T>>(state_.momentum);
                auto& vel = std::get<std::vector<T>>(state_.velocity);

                out.write(reinterpret_cast<const char*>(&param.data()), param.size() * sizeof(T));
                out.write(reinterpret_cast<const char*>(mom.data()), param.size() * sizeof(T));
                out.write(reinterpret_cast<const char*>(vel.data()), param.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template class AdaMM<std::tuple<agon::Parameter<float>>>;
    template class AdaMM<std::tuple<agon::Parameter<double>>>;
    template class AdaMM<std::tuple<agon::Parameter<float>, agon::Parameter<double>>>;
    template class AdaMM<std::tuple<agon::Parameter<double>, agon::Parameter<float>>>;
}