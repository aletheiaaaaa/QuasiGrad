#pragma once

#include "../parameter.hpp"

#include <algorithm>
#include <cstdint>

namespace agon::optim {
  struct OptimizerState {
    int64_t step = 0;
  };

  template<typename... Ts>
  class Optimizer {
    public:
      explicit Optimizer(ParameterPack<Ts...> parameters) : parameters_(parameters) {}

      void zero_grad() {
        std::apply([](auto&... param_vecs) {
          (std::ranges::for_each(param_vecs, [](auto& param_ref) {
            auto& param = param_ref.get();
            param.zero_grad();
          }), ...);
        }, parameters_.data);
      }

      template<typename T>
        requires std::derived_from<T, Parameter<typename T::DataType>>
      void add_parameter(T& param) {
        parameters_.add_parameter(param);
      }

      virtual bool recompute() const { return false; }
      virtual bool use_ref() const { return false; }

      virtual void step() = 0;

      virtual void load_from_bin(const std::string& path) = 0;
      virtual void save_to_bin(const std::string& path) const = 0;

      ~Optimizer() = default;
    protected:
      OptimizerState state_;
      ParameterPack<Ts...> parameters_;
  };
}
