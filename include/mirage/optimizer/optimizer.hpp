#pragma once

#include "../parameter.hpp"

#include <cstdint>
#include <sstream>

namespace mirage::optim {
  struct OptimizerState {
    int64_t step = 0;
  };

  template<typename DedupedPack>
    requires detail::NonConstPack<DedupedPack>
  class Optimizer {
    public:
      explicit Optimizer(ParameterPack<DedupedPack> parameters) : parameters_(parameters) {}

      void zero_grad() {
        std::apply([](auto&... param_vecs) {
          (std::ranges::for_each(param_vecs, [](auto& param_ref) {
            auto& param = param_ref.get();
            param.zero_grad();
          }), ...);
        }, parameters_.data);
      }

      virtual bool recompute() const { return false; }
      virtual bool use_ref() const { return false; }

      virtual void step() = 0;

      virtual void load_from_bin(const std::string& path) = 0;
      virtual void save_to_bin(const std::string& path) const = 0;

      virtual std::string optimizer_type() const = 0;

      ~Optimizer() = default;

    protected:
      OptimizerState state_;
      ParameterPack<DedupedPack> parameters_;

      void handle_type_error(const std::string& recieved) const {
        const std::string expected = optimizer_type();
        std::ostringstream err;

        auto get_name = [](const std::string& type) {
          auto delimit = type.find('<');
          return type.substr(0, delimit);
        };

        auto get_body = [](const std::string& type) {
          auto open = type.find('<');
          return type.substr(open + 1, type.length() - open - 2);
        };

        auto get_types = [](const std::string& body) {
          std::vector<std::string> types;
          size_t depth = 0, start = 0;

          for (size_t i = 0; i < body.size(); ++i) {
            if (body[i] == '<') ++depth;
            else if (body[i] == '>') --depth;
            else if (body[i] == ',' && depth == 0) {
              auto token = body.substr(start, i - start);
              if (!token.empty() && token.front() == ' ') token.erase(0, 1);
              types.push_back(token);
              start = i + 1;
            }
          }

          auto token = body.substr(start);
          if (!token.empty() && token.front() == ' ') token.erase(0, 1);
          if (!token.empty()) types.push_back(token);

          return types;
        };

        auto get_type_name = [](const std::string& group) {
          auto bracket = group.find('[');
          return bracket != std::string::npos ? group.substr(0, bracket) : group;
        };

        auto get_shapes = [](const std::string& group) -> std::vector<std::string> {
          auto open = group.find('[');
          auto close = group.rfind(']');
          if (open == std::string::npos || close == std::string::npos) return {};

          std::vector<std::string> shapes;
          std::string inner = group.substr(open + 1, close - open - 1);
          size_t start = 0;

          for (size_t i = 0; i <= inner.size(); ++i) {
            if (i == inner.size() || inner[i] == ',') {
              shapes.push_back(inner.substr(start, i - start));
              start = i + 1;
            }
          }

          return shapes;
        };

        std::string exp_name = get_name(expected);
        std::string got_name = get_name(recieved);

        if (exp_name != got_name) {
          err << "Optimizer name differs: expected '" << exp_name << "', got '" << got_name << "'";
          throw std::runtime_error(err.str());
        }

        auto exp_types = get_types(get_body(expected));
        auto got_types = get_types(get_body(recieved));

        if (exp_types.size() != got_types.size()) {
          err << "Number of parameter types differs: expected " << exp_types.size() << ", got " << got_types.size();
          throw std::runtime_error(err.str());
        }

        for (size_t i = 0; i < exp_types.size(); ++i) {
          if (exp_types[i] == got_types[i]) continue;

          std::string exp_tn = get_type_name(exp_types[i]);
          std::string got_tn = get_type_name(got_types[i]);

          if (exp_tn != got_tn) {
            err << "Parameter group " << i << " dtype differs: expected '" << exp_tn << "', got '" << got_tn << "'";
            throw std::runtime_error(err.str());
          }

          auto exp_shapes = get_shapes(exp_types[i]);
          auto got_shapes = get_shapes(got_types[i]);

          if (exp_shapes.size() != got_shapes.size()) {
            err << "Parameter group " << i << " (" << exp_tn << ") count differs: expected " << exp_shapes.size() << " parameters, got " << got_shapes.size();
            throw std::runtime_error(err.str());
          }

          for (size_t j = 0; j < exp_shapes.size(); ++j) {
            if (exp_shapes[j] != got_shapes[j]) {
              err << "Parameter group " << i << " (" << exp_tn << ") parameter " << j << " shape differs: expected " << exp_shapes[j] << ", got " << got_shapes[j];
              throw std::runtime_error(err.str());
            }
          }
        }

        throw std::runtime_error("Optimizer type mismatch: expected " + expected + ", got " + recieved);
      }
  };
}
