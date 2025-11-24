#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <typeinfo>

namespace agon {
    class IParameter {
        public:
            virtual ~IParameter() = default;

            virtual void* grad_ptr() = 0;
            virtual const void* grad_ptr() const = 0;

            virtual void* data_ptr() = 0;
            virtual const void* data_ptr() const = 0;

            virtual const std::type_info& dtype() const = 0;

            virtual size_t size() const = 0;
            virtual void zero_grad() = 0;
    };

    template<typename T>
    class Parameter : public IParameter {
        public:
            Parameter(size_t size);
            Parameter(const std::vector<T>& data);
            template<size_t N>
            Parameter(const std::array<T, N>& data);

            std::vector<T>& grad();
            const std::vector<T>& grad() const;

            std::vector<T>& data();
            const std::vector<T>& data() const;

            void* grad_ptr() override;
            const void* grad_ptr() const override;

            void* data_ptr() override;
            const void* data_ptr() const override;

            const std::type_info& dtype() const override;

            size_t size() const override;

            void zero_grad() override;

            void accumulate(const std::vector<T>& new_grad);
            void update(const std::vector<T>& new_val);
        private:
            std::vector<T> data_;
            std::vector<T> grad_;
    };

    template<typename Q, typename T>
    class Quantized : public Parameter<T> {
        public:
            Quantized(size_t size);

            Quantized(const std::vector<Q>& data, float scale, float zero_point);
            template<size_t N>
            Quantized(const std::array<Q, N>& data, float scale, float zero_point);

            Quantized(const std::vector<T>& data, bool use_affine = false, bool use_full = false);
            template<size_t N>
            Quantized(const std::array<T, N>& data, bool use_affine = false, bool use_full = false);

            std::vector<Q> quantized() const;
            std::vector<T> fake_quantized() const;

            float scale() const;
            float zero_point() const;

            const std::type_info& quantized_dtype() const;

        private:
            float scale_ = 1.0f;
            float zero_point_ = 0.0f;

            bool use_affine_ = false;
            bool use_full_ = false;

            std::pair<float, float> compute_quant_params_(T min, T max, bool use_affine, bool use_full) const;
    };
}