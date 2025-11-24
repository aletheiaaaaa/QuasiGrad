#include "../include/agon/parameter.h"
#include "../include/agon/detail/simd/ops.h"
#include "../include/agon/detail/simd/utils.h"

namespace agon {
    template<typename T>
    Parameter<T>::Parameter(size_t size) : data_(size), grad_(size, T(0)) {};

    template<typename T>
    template<size_t N>
    Parameter<T>::Parameter(const std::array<T, N>& data) : data_(data), grad_(data.size(), T(0)) {};

    template<typename T>
    Parameter<T>::Parameter(const std::vector<T>& data) : data_(data), grad_(data.size(), T(0)) {};

    template<typename T>
    std::vector<T>& Parameter<T>::grad() {
        return grad_;
    }

    template<typename T>
    const std::vector<T>& Parameter<T>::grad() const {
        return grad_;
    }

    template<typename T>
    std::vector<T>& Parameter<T>::data() {
        return data_;
    }

    template<typename T>
    const std::vector<T>& Parameter<T>::data() const {
        return data_;
    }

    template<typename T>
    void* Parameter<T>::grad_ptr() {
        return static_cast<void*>(grad_.data());
    }

    template<typename T>
    const void* Parameter<T>::grad_ptr() const {
        return static_cast<const void*>(grad_.data());
    }

    template<typename T>
    void* Parameter<T>::data_ptr() {
        return static_cast<void*>(data_.data());
    }

    template<typename T>
    const void* Parameter<T>::data_ptr() const {
        return static_cast<const void*>(data_.data());
    }

    template<typename T>
    const std::type_info& Parameter<T>::dtype() const {
        return typeid(T);
    }

    template<typename T>
    size_t Parameter<T>::size() const {
        return data_.size();
    }

    template<typename T>
    void Parameter<T>::zero_grad() {
        std::fill(grad_.begin(), grad_.end(), T(0));
    }

    template<typename T>
    void Parameter<T>::accumulate(const std::vector<T>& new_grad) {
        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= grad_.size(); i += vec_size * unroll_factor) {
            simd::unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset = index * vec_size;

                auto grad_vec = simd::load<simd::vec<T>>(&grad_[i + offset]);
                auto new_vec = simd::load<simd::vec<T>>(&new_grad[i + offset]);
                grad_vec = simd::add(grad_vec, new_vec);
                simd::store(&grad_[i + offset], grad_vec);
            });
        }

        for (; i < grad_.size(); ++i) {
            grad_[i] += new_grad[i];
        }
    }

    template<typename T>
    void Parameter<T>::update(const std::vector<T>& new_val) {
        data_ = new_val;
    }

    template<typename Q, typename T>
    Quantized<Q, T>::Quantized(size_t size) : Parameter<T>(size), scale_(1.0f), zero_point_(0.0f) {}

    template<typename Q, typename T>
    Quantized<Q, T>::Quantized(const std::vector<Q>& data, float scale, float zero_point)
        : Parameter<T>(data.size()), scale_(scale), zero_point_(zero_point) {

        auto& vals = this->data();

        T scale_cast = static_cast<T>(scale);
        T zero_point_cast = static_cast<T>(zero_point);

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= data.size(); i += vec_size * unroll_factor) {
            simd::unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset = index * vec_size;

                auto q_vec = simd::load<simd::vec<Q>>(&data[i + offset]);
                auto q_float_vec = simd::cast<simd::vec<T>>(q_vec);
                auto scale_vec = simd::set1<simd::vec<T>>(scale_cast);
                auto zero_point_vec = simd::set1<simd::vec<T>>(zero_point_cast);

                auto val_vec = simd::sub(q_float_vec, zero_point_vec);
                val_vec = simd::mul(val_vec, scale_vec);

                simd::store(&vals[i + offset], val_vec);
            });
        }

        for (; i < data.size(); ++i) {
            vals[i] = scale_cast * (static_cast<T>(data[i]) - zero_point_cast);
        }
    }

    template<typename Q, typename T>
    template<size_t N>
    Quantized<Q, T>::Quantized(const std::array<Q, N>& data, float scale, float zero_point)
        : Parameter<T>(data.size()), scale_(scale), zero_point_(zero_point) {

        auto& vals = this->data();

        T scale_cast = static_cast<T>(scale);
        T zero_point_cast = static_cast<T>(zero_point);

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= data.size(); i += vec_size * unroll_factor) {
            simd::unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset = index * vec_size;

                auto q_vec = simd::load<simd::vec<Q>>(&data[i + offset]);
                auto q_float_vec = simd::cast<simd::vec<T>>(q_vec);
                auto scale_vec = simd::set1<simd::vec<T>>(scale_cast);
                auto zero_point_vec = simd::set1<simd::vec<T>>(zero_point_cast);

                auto val_vec = simd::sub(q_float_vec, zero_point_vec);
                val_vec = simd::mul(val_vec, scale_vec);
                simd::store(&vals[i + offset], val_vec);
            });
        }

        for (; i < data.size(); ++i) {
            vals[i] = scale_cast * (static_cast<T>(data[i]) - zero_point_cast);
        }
    }

    template<typename Q, typename T>
    std::vector<Q> Quantized<Q, T>::quantized() const {
        const auto& vals = this->data();
        std::vector<Q> quantized_data(vals.size());

        T inv_scale = static_cast<T>(1.0f / scale_);
        T zero_point_cast = static_cast<T>(zero_point_);

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
            simd::unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset = index * vec_size;

                auto val_vec = simd::load<simd::vec<T>>(&vals[i + offset]);
                auto inv_scale_vec = simd::set1<simd::vec<T>>(inv_scale);
                auto zero_point_vec = simd::set1<simd::vec<T>>(zero_point_cast);
                auto q_vec = simd::fmadd(val_vec, inv_scale_vec, zero_point_vec);

                if constexpr (simd::IsUpcast<T, Q>) {
                    constexpr size_t mult = simd::vec<T>::size / simd::vec<Q>::size;
                    simd::unroll<mult>([&]<size_t j>() {
                        constexpr size_t q_offset = j * simd::vec<Q>::size;

                        auto q_subvec = simd::cast<simd::vec<Q>, j>(q_vec);
                        simd::store(&quantized_data[i + offset + q_offset], q_subvec);
                    });
                } else {
                    q_vec = simd::cast<simd::vec<Q>>(q_vec);
                    simd::store(&quantized_data[i + offset], q_vec);
                }
            });
        }

        for (; i < vals.size(); ++i) {
            quantized_data[i] = static_cast<Q>(inv_scale * vals[i] + zero_point_cast);
        }

        return quantized_data;
    }

    template<typename Q, typename T>
    std::vector<T> Quantized<Q, T>::fake_quantized() const {
        const auto& vals = this->data();
        std::vector<T> fake_quantized_data(vals.size());

        T inv_scale = static_cast<T>(1.0f / scale_);
        T zero_point_cast = static_cast<T>(zero_point_);

        constexpr size_t vec_size = simd::vec<T>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= vals.size(); i += vec_size * unroll_factor) {
            simd::unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset = index * vec_size;
                
                auto val_vec = simd::load<simd::vec<T>>(&vals[i + offset]);
                auto scale_vec = simd::set1<simd::vec<T>>(scale_);
                auto inv_scale_vec = simd::set1<simd::vec<T>>(inv_scale);
                auto zero_point_vec = simd::set1<simd::vec<T>>(zero_point_cast);

                auto q_vec = simd::fmadd(val_vec, inv_scale_vec, zero_point_vec);
                q_vec = simd::round(q_vec);

                auto dq_vec = simd::sub(q_vec, zero_point_vec);
                dq_vec = simd::mul(dq_vec, scale_vec);
                simd::store(&fake_quantized_data[i + offset], dq_vec);
            });
        }

        for (; i < vals.size(); ++i) {
            T q = std::round(inv_scale * vals[i] + zero_point_cast);
            fake_quantized_data[i] = scale_ * (q - zero_point_cast);
        }

        return fake_quantized_data;
    }

    template<typename Q, typename T>
    float Quantized<Q, T>::scale() const {
        return scale_;
    }

    template<typename Q, typename T>
    float Quantized<Q, T>::zero_point() const {
        return zero_point_;
    }

    template<typename Q, typename T>
    const std::type_info& Quantized<Q, T>::quantized_dtype() const {
        return typeid(Q);
    }
}