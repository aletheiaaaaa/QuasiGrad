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

            virtual const std::type_info& grad_type() const = 0;
            virtual const std::type_info& data_type() const = 0;

            virtual size_t size() const = 0;
            virtual void zero_grad() = 0;
    };

    template<typename T, typename G = float>
    class Parameter : public IParameter {
        public:
            Parameter(size_t size);
            Parameter(const std::vector<T>& data);
            template<size_t N>
            Parameter(const std::array<T, N>& data);

            std::vector<G>& grad();
            const std::vector<G>& grad() const;

            std::vector<T>& data();
            const std::vector<T>& data() const;

            void* grad_ptr() override;
            const void* grad_ptr() const override;

            void* data_ptr() override;
            const void* data_ptr() const override;

            const std::type_info& grad_type() const override;
            const std::type_info& data_type() const override;

            size_t size() const override;

            void zero_grad() override;

            void accumulate(const std::vector<G>& new_grad);
            void update(const std::vector<T>& new_val);
        private:
            std::vector<T> vals;
            std::vector<G> grads;
    };
}