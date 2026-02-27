#include "agon/optimizers/impls/adam.h"

#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

#include <fstream>
#include <filesystem>

namespace agon::optim {
    template<typename... Ts>
    void Adam<Ts...>::step() {
        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

                auto& grad_full = param.grad();
                auto& data_full = param.data();
                auto& mom_full = std::get<std::vector<T>>(state_.momentum);
                auto& vel_full = std::get<std::vector<T>>(state_.velocity);

                constexpr size_t vec_size = simd::vec<T>::size;
                constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                size_t i = 0;
                for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
                    simd::unroll<unroll_factor>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad = simd::load<T>(&grad_full[i + offset]);
                        auto mom = simd::load<T>(&mom_full[i + offset]);
                        auto vel = simd::load<T>(&vel_full[i + offset]);

                        if (options_.maximize) grad = simd::neg(grad);

                        auto beta1 = simd::set1<T>(options_.beta1);
                        mom = simd::fmadd(beta1, mom, grad);
                        mom = simd::fnmadd(beta1, grad, mom);

                        auto beta2 = simd::set1<T>(options_.beta2);
                        auto grad_squared = (options_.use_adazo) ? simd::mul(mom, mom) : simd::mul(grad, grad);
                        vel = simd::fmadd(beta2, vel, grad_squared);
                        vel = simd::fnmadd(beta2, grad_squared, vel);

                        simd::store(&mom[i + offset], mom);
                        simd::store(&vel[i + offset], vel);

                        auto epsilon = simd::set1<T>(options_.epsilon);
                        auto update = simd::div(mom, simd::add(simd::sqrt(vel), epsilon));

                        auto lr = simd::set1<T>(options_.lr);
                        auto data = simd::load<T>(&data_full[i + offset]);
                        data = simd::fmadd(lr, update, data);
                        simd::store(&data_full[i + offset], data);
                    });
                }

                for (; i < grad_full.size(); ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];

                    T mom = options_.beta1 * mom_full[i] + (1 - options_.beta1) * grad;
                    T vel = options_.beta2 * vel_full[i] + (1 - options_.beta2) * grad * grad;

                    mom_full[i] = mom;
                    vel_full[i] = vel;

                    T update = mom / (std::sqrt(vel) + options_.epsilon);
                    data_full[i] += options_.lr * update;
                }
            }), ...);
        }, this->parameters_.data);

        state_.step++;
    }

    template<typename... Ts>
    void Adam<Ts...>::load_from_bin(const std::string& path_str) {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
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
    void Adam<Ts...>::save_to_bin(const std::string& path_str) const {
        std::filesystem::path path(path_str);
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
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

    template class Adam<std::tuple<agon::Parameter<float>>>;
    template class Adam<std::tuple<agon::Parameter<double>>>;
    template class Adam<std::tuple<agon::Parameter<float>, agon::Parameter<double>>>;
    template class Adam<std::tuple<agon::Parameter<double>, agon::Parameter<float>>>;
}