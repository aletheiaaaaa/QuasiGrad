#include "agon/optimizers/impls/sgd.h"

#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

#include <fstream>
#include <filesystem>

namespace agon::optim {
    template<typename... Ts>
    void SGD<Ts...>::step() {
        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

                auto& grad_full = param.grad();
                auto& data_full = param.data();
                auto& mom_full = std::get<std::vector<T>>(state_.momenta);

                constexpr size_t vec_size = simd::vec<T>::size;
                constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                size_t i = 0;
                for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
                    simd::unroll<unroll_factor>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad = simd::load<T>(&grad_full[i + offset]);
                        auto mom = simd::load<T>(&mom_full[i + offset]);

                        if (options_.maximize) grad = simd::neg(grad);

                        auto mom_coeff = simd::set1<T>(options_.momentum);
                        mom = simd::fmadd(mom_coeff, mom, grad);
                        simd::store(&mom_full[i + offset], mom);

                        if (options_.nesterov) mom = simd::fmadd(mom_coeff, mom, grad);

                        auto lr = simd::set1<T>(options_.lr);

                        auto data = simd::load<T>(&data_full[i + offset]);
                        data = simd::fmadd(lr, mom, data);
                        simd::store(&data_full[i + offset], data);
                    });
                }

                for (; i < grad_full.size(); ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                    T mom = options_.momentum * mom_full[i] + grad;
                    mom_full[i] = mom;

                    T update = options_.nesterov ? (options_.momentum * mom + grad) : mom;
                    data_full[i] += options_.lr * update;
                }
            }), ...);
        }, this->parameters_.data);

        state_.step++;
    }

    template<typename... Ts>
    void SGD<Ts...>::load_from_bin(const std::string& path_str) {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open file: " + path_str);

        file.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& mom = std::get<std::vector<T>>(state_.momenta);

                file.read(reinterpret_cast<char*>(&param.data()), param.size() * sizeof(T));
                file.read(reinterpret_cast<char*>(mom.data()), param.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template<typename... Ts>
    void SGD<Ts...>::save_to_bin(const std::string& path_str) const {
        std::filesystem::path path(path_str);
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open file: " + path_str);

        file.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& mom = std::get<std::vector<T>>(state_.momenta);

                file.write(reinterpret_cast<const char*>(&param.data()), param.size() * sizeof(T));
                file.write(reinterpret_cast<const char*>(mom.data()), param.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template class SGD<std::tuple<agon::Parameter<float>>>;
    template class SGD<std::tuple<agon::Parameter<double>>>;
    template class SGD<std::tuple<agon::Parameter<float>, agon::Parameter<double>>>;
    template class SGD<std::tuple<agon::Parameter<double>, agon::Parameter<float>>>;
}
