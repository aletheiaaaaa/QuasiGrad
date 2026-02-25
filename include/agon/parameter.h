#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <typeinfo>
#include <span>
#include <functional>

#include "detail/dedup.h"

namespace agon {
    template<typename T>
        requires std::is_floating_point_v<T>
    class Parameter {
        public:
            using DataType = T;

            Parameter(size_t size);
            Parameter(const std::span<T>& data);

            std::vector<T>& grad();
            const std::vector<T>& grad() const;

            std::vector<T>& data();
            const std::vector<T>& data() const;

            size_t size() const;

            void zero_grad();

            void accumulate(const std::vector<T>& new_grad);
            void update(const std::vector<T>& new_val);

        private:
            std::vector<T> data_;
            std::vector<T> grad_;
    };

    template<typename Q, typename T>
        requires (std::is_same_v<Q, int16_t> || std::is_same_v<Q, int8_t>) && std::is_floating_point_v<T>
    class Quantized : public Parameter<T> {
        public:
            using QuantizedType = Q;

            Quantized(size_t size);
            Quantized(const std::span<Q>& data, float scale = 1.0f, float zero_point = 0.0f);

            std::vector<Q> quantized() const;
            std::vector<T> fake_quantized() const;

            float scale() const;
            float zero_point() const;

        private:
            float scale_ = 1.0f;
            float zero_point_ = 0.0f;
    };

    template<typename T>
    using AsParameter_t = Parameter<typename T::DataType>;

    template<typename T>
    using RefVec = std::vector<std::reference_wrapper<T>>;

    template<typename DedupedTuple>
    struct ParameterPack {
        dedup::TransformTuple_t<RefVec, DedupedTuple> data{};

        template<typename... Ts>
            requires (std::derived_from<Ts, Parameter<typename Ts::DataType>> && ...)
        ParameterPack(Ts&... params) {
            (std::get<RefVec<AsParameter_t<Ts>>>(data)
                .emplace_back(static_cast<AsParameter_t<Ts>&>(params)), ...);
        }

        template<typename T>
            requires std::derived_from<T, Parameter<typename T::DataType>>
        void add_parameter(T& param) {
            std::get<RefVec<AsParameter_t<T>>>(data)
                .emplace_back(static_cast<AsParameter_t<T>&>(param));
        }
    };
    template<typename... Ts>
    ParameterPack(Ts&...) -> ParameterPack<dedup::DeduplicatedPack_t<AsParameter_t<std::decay_t<Ts>>...>>;

    template<typename T>
    struct ExtractType {};
    template<typename T>
    struct ExtractType<Parameter<T>> {
        using Type = T;
    };
    template<typename T>
    using ExtractType_t = typename ExtractType<T>::Type;

    extern template class Parameter<float>;
    extern template class Parameter<double>;

    extern template class Quantized<int8_t,  float>;
    extern template class Quantized<int16_t, float>;
    extern template class Quantized<int8_t,  double>;
    extern template class Quantized<int16_t, double>;
}