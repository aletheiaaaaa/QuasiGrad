#pragma once 

#include "utils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <eve/conditional.hpp>
#include <eve/arch/cpu/wide.hpp>
#include <eve/module/core.hpp>
#include <eve/module/core/regular/if_else.hpp>
#include <eve/module/core/regular/store.hpp>

namespace mirage::detail {
  namespace matrix {
    template<typename T, typename F>
      requires std::invocable<F, eve::wide<T>>
    inline std::vector<T> triple_matmul(
      std::vector<T> A,
      std::vector<T> B,
      std::vector<T> C,
      size_t M,
      size_t K,
      size_t N,
      size_t P,
      F&& func
    ) {
      std::vector<T> out;
      out.resize(M * P, T(0));

      constexpr size_t x_height = UNROLL_FACTOR;
      constexpr size_t y_height = std::max(1, UNROLL_FACTOR / 2);
      constexpr size_t vec_size = eve::wide<T>::size();
      constexpr size_t arr_size = x_height * y_height;

      for (size_t i = 0; i < M; i += x_height) {
        size_t i_rem = std::min(x_height, M - i);

        for (size_t j = 0; j < N; j += y_height * vec_size) {
          size_t j_rem = std::min(y_height * vec_size, N - j);

          std::array<eve::wide<T>, arr_size> acc0;
          std::fill(acc0.begin(), acc0.end(), eve::wide<T>(T(0)));

          for (size_t k = 0; k < K; ++k) {
            std::array<eve::wide<T>, x_height> a_tile;
            std::array<eve::wide<T>, y_height> b_tile;

            unroll<x_height>([&]<size_t idx>() {
              a_tile[idx] = (idx < i_rem)
                ? eve::wide<T>(A[(i + idx) * K + k])
                : eve::wide<T>(T(0));
            });

            unroll<y_height>([&]<size_t idx>() {
              size_t valid = std::min(vec_size, j_rem - idx * vec_size);

              b_tile[idx] = (idx * vec_size < j_rem)
                ? eve::if_else(eve::keep_first(valid),  eve::wide<T>(&B[k * N + j + vec_size * idx]), eve::zero)
                : eve::wide<T>(T(0));
            });

            unroll<arr_size>([&]<size_t idx>() {
              constexpr size_t row = idx % x_height;
              constexpr size_t col = idx / x_height;

              acc0[idx] = eve::fma(a_tile[row], b_tile[col], acc0[idx]);
            });
          }

          std::array<T, x_height * y_height * vec_size> temp;
          unroll<arr_size>([&]<size_t idx>() {
            eve::store(acc0[idx], &temp[idx * vec_size]);
          });

          for (size_t k = 0; k < P; k += y_height * vec_size) {
            size_t k_rem = std::min(y_height * vec_size, P - k);

            std::array<eve::wide<T>, arr_size> acc1;
            std::fill(acc1.begin(), acc1.end(), eve::wide<T>(T(0)));

            for (size_t l = 0; l < j_rem; ++l) {
              std::array<eve::wide<T>, x_height> t_tile;
              std::array<eve::wide<T>, y_height> c_tile;

              unroll<x_height>([&]<size_t idx>() {
                size_t row = l % vec_size;
                size_t col = l / vec_size;

                t_tile[idx] = eve::wide<T>(temp[(col * x_height + idx) * vec_size + row]);
              });

              unroll<y_height>([&]<size_t idx>() {
                size_t valid = std::min(vec_size, k_rem - idx * vec_size);

                c_tile[idx] = (idx * vec_size < k_rem)
                  ? eve::if_else(eve::keep_first(valid), eve::wide<T>(&C[(j + l) * P + k + idx * vec_size]), eve::zero)
                  : eve::wide<T>(T(0));
              });

              unroll<arr_size>([&]<size_t idx>() {
                constexpr size_t row = idx % x_height;
                constexpr size_t col = idx / x_height;

                acc1[idx] = eve::fma(t_tile[row], c_tile[col], acc1[idx]);
              });
            }

            unroll<arr_size>([&]<size_t idx>() {
              constexpr size_t row = idx % x_height;
              constexpr size_t col = idx / x_height;

              if (row < i_rem && col * vec_size < k_rem) {
                auto mask = eve::keep_first(std::min(vec_size, k_rem - col * vec_size));

                eve::wide<T> prev = eve::if_else(mask, eve::wide<T>(&out[(i + row) * P + k + col * vec_size]), eve::zero);
                eve::store[mask](prev + acc1[idx], &out[(i + row) * P + k + col * vec_size]);
              }
            });
          }
        }

        for (size_t j = 0; j < P; j += y_height * vec_size) {
          size_t j_rem = std::min(y_height * vec_size, P - j);
          unroll<arr_size>([&]<size_t idx>() {
            constexpr size_t row = idx % x_height;
            constexpr size_t col = idx / x_height;

            if (row < i_rem && col * vec_size < j_rem) {
              auto mask = eve::keep_first(std::min(vec_size, j_rem - col * vec_size));

              eve::wide<T> val = eve::if_else(mask, eve::wide<T>(&out[(i + row) * P + j + col * vec_size]), eve::zero);
              eve::store[mask](func(val), &out[(i + row) * P + j + col * vec_size]);
            }
          });
        }
      }

      return out;
    }
  }

  template<typename T>
  std::vector<T> triple_matmul(
    std::vector<T> A, 
    std::vector<T> B, 
    std::vector<T> C, 
    size_t M, 
    size_t K, 
    size_t N, 
    size_t P
  ) {
    return matrix::triple_matmul(A, B, C, M, K, N, P, [&](eve::wide<T>& reg) {});
  }

  template<typename T>
  std::vector<T> triple_matmul_sign(
    std::vector<T> A, 
    std::vector<T> B, 
    std::vector<T> C, 
    size_t M, 
    size_t K, 
    size_t N, 
    size_t P
  ) {
    return matrix::triple_matmul(A, B, C, M, K, N, P, [&](eve::wide<T>& reg) { return eve::sign(reg); });
  }

  template<typename T>
  std::vector<T> transpose_ema(
    std::vector<T> X_og,
    std::vector<T> X_tp,
    std::vector<T> E,
    size_t M, 
    size_t N, 
    float ema_wt
  ) {
    constexpr size_t x_height = UNROLL_FACTOR;
    constexpr size_t y_height = std::max(1, UNROLL_FACTOR / 2);
    constexpr size_t vec_size = eve::wide<T>::size();
    constexpr size_t arr_size = x_height * y_height;

    for (size_t i = 0; i < M; i += x_height) {
      size_t i_rem = std::min(x_height, M - i);

      for (size_t j = 0; j < M; j += y_height * vec_size) {
        size_t j_rem = std::min(y_height * vec_size, M - j);

        std::array<eve::wide<T>, arr_size> acc;
        std::fill(acc.begin(), acc.end(), eve::wide<T>(T(0)));

        for (size_t k = 0; k < N; ++k) {
          std::array<eve::wide<T>, x_height> og_tile;
          std::array<eve::wide<T>, y_height> tp_tile;

          unroll<x_height>([&]<size_t idx>() {
            og_tile[idx] = (idx < i_rem)
              ? eve::wide<T>(X_og[(i + idx) * N + k])
              : eve::wide<T>(T(0));
          });

          unroll<y_height>([&]<size_t idx>() {
            size_t valid = std::min(vec_size, j_rem - idx * vec_size);

            tp_tile[idx] = (idx * vec_size < j_rem)
              ? eve::if_else(eve::keep_first(valid), eve::wide<T>(&X_tp[k * M + j + idx * vec_size]), eve::zero)
              : eve::wide<T>(T(0));
          });

          unroll<arr_size>([&]<size_t idx>() {
            constexpr size_t row = idx % x_height;
            constexpr size_t col = idx / x_height;

            acc[idx] = eve::fma(og_tile[row], tp_tile[col], acc[idx]);
          });
        }

        unroll<arr_size>([&]<size_t idx>() {
          constexpr size_t row = idx % x_height;
          constexpr size_t col = idx / x_height;

          if (row < i_rem && col * vec_size < j_rem) {
            auto mask = eve::keep_first(std::min(vec_size, j_rem - col * vec_size));

            eve::wide<T> ema = eve::if_else(mask, eve::wide<T>(&E[(i + row) * M + j + col * vec_size]), eve::zero);
            eve::wide<T> wt(ema_wt);

            ema = eve::fma(wt, ema, acc[idx]);
            ema = eve::fnma(wt, acc[idx], ema);

            eve::store[mask](ema, &E[(i + row) * M + j + col * vec_size]);
          }
        });
      }
    }
  }
}