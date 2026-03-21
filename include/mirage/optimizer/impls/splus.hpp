#pragma once

#include "../optimizer.hpp"
#include "../../detail/utils.hpp"

#include <cstddef>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace mirage::optim {
  struct SPlusOptions {
    float lr = 0.5f;
    float beta1 = 0.9f;
    float beta2 = 0.4f;
    float beta3 = 0.999f;
    float lambda = 0.0f;
    int decompose_every = 64;

    bool maximize = false;

    template<class Archive>
    void serialize(Archive& ar) {
      ar(lr, beta1, beta2, beta3, lambda, decompose_every, maximize);
    }
  };

  template<typename DedupedTuple>
  struct SPlusState : public OptimizerState {
    ExtractedVector<DedupedTuple> momentum{};
    ExtractedVector<DedupedTuple> left_velocity{};
    ExtractedVector<DedupedTuple> right_velocity{};
    ExtractedVector<DedupedTuple> left_eigenvectors{};
    ExtractedVector<DedupedTuple> right_eigenvectors{};
    ExtractedVector<DedupedTuple> param_ema{};
  };

  template<typename DedupedTuple>
  class SPlus : public Optimizer<DedupedTuple> {
    public:
      explicit SPlus(ParameterPack<DedupedTuple> parameters, SPlusOptions options = {}, int num_proc = 1)
        : Optimizer<DedupedTuple>(parameters), options_(options), num_proc_(num_proc) {
          assert(std::apply([](auto&... param_vecs) {
            return (std::all_of(param_vecs.begin(), param_vecs.end(),
              [](auto& p) { return p.get().rank() >= 2; }) && ...
            );
          }, this->parameters_.data) && "All parameters must have at least 2 dimensions");

          std::apply([&](auto&... param_vecs) {
            ([&](auto& param_vec) {
              using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
              auto& mom = std::get<ExtractType_t<ParamType>>(this->state_.momentum);
              auto& lvel = std::get<ExtractType_t<ParamType>>(this->state_.left_velocity);
              auto& rvel = std::get<ExtractType_t<ParamType>>(this->state_.right_velocity);
              auto& leig = std::get<ExtractType_t<ParamType>>(this->state_.left_eigenvectors);
              auto& reig = std::get<ExtractType_t<ParamType>>(this->state_.right_eigenvectors);
              auto& ema = std::get<ExtractType_t<ParamType>>(this->state_.param_ema);
              for (auto& param_ref : param_vec) {
                auto& param = param_ref.get();
                using T = typename ParamType::DataType;
                mom.insert(mom.end(), param.numel(), T(0));
                lvel.insert(lvel.end(), param.numel(), T(0));
                rvel.insert(rvel.end(), param.numel(), T(0));
                leig.insert(leig.end(), param.numel(), T(0));
                reig.insert(reig.end(), param.numel(), T(0));
                ema.insert(ema.end(), param.numel(), T(0));
              }
            }(param_vecs), ...);
          }, this->parameters_.data);
        }

      void step() override {
        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<ExtractType_t<ParamType>>(state_.momentum);
            auto& lvel_full = std::get<ExtractType_t<ParamType>>(state_.left_velocity);
            auto& rvel_full = std::get<ExtractType_t<ParamType>>(state_.right_velocity);
            auto& ema_full = std::get<ExtractType_t<ParamType>>(state_.param_ema);

            size_t state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr size_t vec_size = eve::wide<T>::size();

              std::vector<std::thread> threads;
              size_t chunk_size = (param.numel() + num_proc_ - 1) / num_proc_;

              for (size_t t = 0; t < num_proc_; ++t) {
                threads.emplace_back([&, t]() {
                  size_t start = t * chunk_size;
                  size_t end = std::min(start + chunk_size, param.numel());

                  size_t i = start;

                  // TODO: matmuls
                });
              }

              for (auto& thread : threads) {
                thread.join();
              }
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
        this->state_.step++;
      }

      void load_from_bin(const std::string& path_str) override {
        std::filesystem::path path(path_str);
        path.replace_extension(".bin");

        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        cereal::BinaryInputArchive ar(in);
        std::string name;
        ar(name);
        if (name != optimizer_type()) this->handle_type_error(name);

        ar(options_, state_.step, state_.momentum, state_.left_velocity, state_.right_velocity, state_.param_ema);

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

      void save_to_bin(const std::string& path_str) const override {
        std::filesystem::path path(path_str);
        path.replace_extension(".bin");

        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        cereal::BinaryOutputArchive ar(out);
        std::string name(optimizer_type());

        ar(name, options_, state_.step, state_.momentum, state_.left_velocity, state_.right_velocity, state_.param_ema);

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

      std::string optimizer_type() const override {
        std::string type = "SPlus<";
        bool first = true;

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;

            if (!first) type += ", ";
            first = false;
            type += PrintType<ParamType>::name() + "[";

            bool pfirst = true;
            for (auto& param_ref : param_vec) {
              if (!pfirst) type += ",";
              pfirst = false;

              auto& shape = param_ref.get().size();
              for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) type += "x";
                type += std::to_string(shape[i]);
              }
            }

            type += "]";
          }(param_vecs), ...);
        }, this->parameters_.data);

        type += ">";
        return type;
      }

    private:
      SPlusOptions options_;
      SPlusState<DedupedTuple> state_;
      int num_proc_ = 1;
  };
}