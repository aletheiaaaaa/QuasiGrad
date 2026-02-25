#include "agon/optimizers/impls/sgd.h"

namespace agon::optim {

    template<typename... Ts>
    void SGD<Ts...>::step() {

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

                auto& grad = param.grad();
                auto& data = param.data();
                auto& mom = std::get<std::vector<T>>(state_.momenta);

                constexpr size_t vec_size = simd::vec<T>::size;
                constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                size_t i = 0;
                for (; i + vec_size * unroll_factor <= grad.size(); i += vec_size * unroll_factor) {
                    simd::unroll<unroll_factor>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad_vec = simd::load<T>(&grad[i + offset]);
                        auto mom_vec = simd::load<T>(&mom[i + offset]);

                        if (options_.maximize) grad_vec = simd::neg(grad_vec);

                        auto mom_coeff = simd::set1<T>(options_.momentum);
                        mom_vec = simd::fmadd(mom_coeff, mom_vec, grad_vec);
                        simd::store(&mom[i + offset], mom_vec);

                        if (options_.nesterov) mom_vec = simd::fmadd(mom_coeff, mom_vec, grad_vec);

                        auto lr_vec = simd::set1<T>(options_.lr);

                        auto data_vec = simd::load<T>(&data[i + offset]);
                        data_vec = simd::fmadd(lr_vec, mom_vec, data_vec);
                        simd::store(&data[i + offset], data_vec);
                    });
                }

                for (; i < grad.size(); ++i) {
                    T grad_val = options_.maximize ? -grad[i] : grad[i];
                    T mom_val = options_.momentum * mom[i] + grad_val;
                    mom[i] = mom_val;

                    T update = options_.nesterov ? (options_.momentum * mom_val + grad_val) : mom_val;
                    data[i] += options_.lr * update;
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
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
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
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
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
}
