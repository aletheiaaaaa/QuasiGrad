#include "agon/optimizers/impls/spider.h"

#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

#include <fstream>
#include <filesystem>

namespace agon::optim {
    template<typename... Ts>
    void Spider<Ts...>::step() {
        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

                auto& grad = param.grad();
                auto& data = param.data();
                auto& prev = std::get<std::vector<T>>(state_.prev_grad);

                constexpr size_t vec_size = simd::vec<T>::size;
                constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                size_t i = 0;
                for (; i + vec_size * unroll_factor <= grad.size(); i += vec_size * unroll_factor) {
                    simd::unroll<unroll_factor>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad_vec = simd::load<T>(&grad[i + offset]);
                        auto prev_grad_vec = simd::load<T>(&prev[i + offset]);

                        if (options_.maximize) grad_vec = simd::neg(grad_vec);

                        auto update_vec = simd::sub(grad_vec, prev_grad_vec);
                        auto data_vec = simd::load<T>(&data[i + offset]);
                        data_vec = simd::add(update_vec, data_vec);

                        simd::store(&data[i + offset], data_vec);
                        simd::store(&prev[i + offset], grad_vec);
                    });
                }

                for (; i < grad.size(); ++i) {
                    T grad_val = options_.maximize ? -grad[i] : grad[i];
                    T update_val = grad_val - prev[i];
                    data[i] += update_val;
                    prev[i] = grad_val;
                }
            }), ...);
        }, this->parameters_.data);

        state_.step++;
    }

    template<typename... Ts>
    void Spider<Ts...>::load_from_bin(const std::string& path_str) {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& prev_grad = std::get<std::vector<T>>(state_.prev_grad);

                in.read(reinterpret_cast<char*>(&param.data()), param.size() * sizeof(T));
                in.read(reinterpret_cast<char*>(prev_grad.data()), prev_grad.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template<typename... Ts>
    void Spider<Ts...>::save_to_bin(const std::string& path_str) const {
        std::filesystem::path path(path_str);
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& prev_grad = std::get<std::vector<T>>(state_.prev_grad);

                out.write(reinterpret_cast<const char*>(&param.data()), param.size() * sizeof(T));
                out.write(reinterpret_cast<const char*>(prev_grad.data()), prev_grad.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template class Spider<std::tuple<Parameter<float>>>;
    template class Spider<std::tuple<Parameter<double>>>;
    template class Spider<std::tuple<Parameter<float>, Parameter<double>>>;
    template class Spider<std::tuple<Parameter<double>, Parameter<float>>>;
}