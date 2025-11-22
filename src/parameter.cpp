#include "../include/agon/parameter.h"
#include "../include/agon/detail/simd/ops.h"
#include "../include/agon/detail/simd/utils.h"

namespace agon {
    template<typename T, typename G>
    Parameter<T, G>::Parameter(size_t size) : vals(size), grads(size, T(0)) {};

    template<typename T, typename G>
    template<size_t N>
    Parameter<T, G>::Parameter(const std::array<T, N>& data) : vals(data), grads(data.size(), T(0)) {};

    template<typename T, typename G>
    Parameter<T, G>::Parameter(const std::vector<T>& data) : vals(data), grads(data.size(), T(0)) {};

    template<typename T, typename G>
    std::vector<G>& Parameter<T, G>::grad() {
        return grads;
    }

    template<typename T, typename G>
    const std::vector<G>& Parameter<T, G>::grad() const {
        return grads;
    }

    template<typename T, typename G>
    std::vector<T>& Parameter<T, G>::data() {
        return vals;
    }

    template<typename T, typename G>
    const std::vector<T>& Parameter<T, G>::data() const {
        return vals;
    }

    template<typename T, typename G>
    void* Parameter<T, G>::grad_ptr() {
        return static_cast<void*>(grads.data());
    }

    template<typename T, typename G>
    const void* Parameter<T, G>::grad_ptr() const {
        return static_cast<const void*>(grads.data());
    }

    template<typename T, typename G>
    void* Parameter<T, G>::data_ptr() {
        return static_cast<void*>(vals.data());
    }

    template<typename T, typename G>
    const void* Parameter<T, G>::data_ptr() const {
        return static_cast<const void*>(vals.data());
    }

    template<typename T, typename G>
    const std::type_info& Parameter<T, G>::grad_type() const {
        return typeid(G);
    }

    template<typename T, typename G>
    const std::type_info& Parameter<T, G>::data_type() const {
        return typeid(T);
    }

    template<typename T, typename G>
    size_t Parameter<T, G>::size() const {
        return vals.size();
    }

    template<typename T, typename G>
    void Parameter<T, G>::zero_grad() {
        std::fill(grads.begin(), grads.end(), T(0));
    }

    template<typename T, typename G>
    void Parameter<T, G>::accumulate(const std::vector<G>& new_grad) {
        constexpr size_t vec_size = vec<G>::size;
        constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

        size_t i = 0;
        for (; i + vec_size * unroll_factor <= grads.size(); i += vec_size * unroll_factor) {
            unroll<unroll_factor>([&]<size_t index>() {
                constexpr size_t offset = index * vec_size;
                
                auto grad_vec = simd::load<vec<G>>(&grads[i + offset]);
                auto new_vec = simd::load<vec<G>>(&new_grad[i + offset]);
                grad_vec = simd::add(grad_vec, new_vec);
                simd::store(&grads[i + offset], grad_vec);
            });
        }

        for (; i < grads.size(); ++i) {
            grads[i] += new_grad[i];
        }
    }

    template<typename T, typename G>
    void Parameter<T, G>::update(const std::vector<T>& new_val) {
        vals = new_val;
    }
}