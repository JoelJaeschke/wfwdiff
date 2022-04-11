#ifndef WFWDIFF_AUTODIFF_H
#define WFWDIFF_AUTODIFF_H

#include <cmath>
#include <concepts>
#include <iostream>
#include <tuple>
#include <utility>

#include "vector.hpp"

namespace wfwdiff {
namespace autodiff {

namespace detail {
template <typename T>
concept Differentiable = std::integral<T> || std::floating_point<T> ||
    wfwdiff::generic_vec::differentiable_vector<T>;
}

template <typename T, typename U>
requires detail::Differentiable<T> && detail::Differentiable<U>
struct var {
  T value;
  U grad;

  constexpr var() : value(), grad(){};
  constexpr var(T value) : value(value), grad(){};
  constexpr var(T value, U grad) : value(value), grad(grad){};

  constexpr var<T, U> operator+(const var<T, U> rhs) const {
    return var<T, U>(value + rhs.value, grad + rhs.grad);
  };

  constexpr var<T, U> operator-(const var<T, U> rhs) const {
    return var<T, U>(value - rhs.value, grad - rhs.grad);
  };

  constexpr var<T, U> operator*(const var<T, U> rhs) const {
    return var<T, U>(value * rhs.value, rhs.grad * value + grad * rhs.value);
  };

  constexpr var<T, U> operator/(const var<T, U> rhs) const {
    return var<T, U>(value / rhs.value, (grad / rhs.value - rhs.grad * value) /
                                            (rhs.val * rhs.val));
  };

  constexpr var<T, U> operator+=(const var<T, U> rhs) const {
    return this + rhs;
  };
  constexpr var<T, U> operator-=(const var<T, U> rhs) const {
    return this - rhs;
  };
  constexpr var<T, U> operator*=(const var<T, U> rhs) const {
    return this * rhs;
  };
  constexpr var<T, U> operator/=(const var<T, U> rhs) const {
    return this / rhs;
  };

  constexpr var<T, U> &operator+=(const var<T, U> rhs) { return this + rhs; };
  constexpr var<T, U> &operator-=(const var<T, U> rhs) { return this - rhs; };
  constexpr var<T, U> &operator*=(const var<T, U> rhs) { return this * rhs; };
  constexpr var<T, U> &operator/=(const var<T, U> rhs) { return this / rhs; };
};

template <typename... Args> struct At {
  std::tuple<Args...> args;
  At(std::tuple<Args...> args) : args(args){};
};

template <typename... Args> struct Wrt {
  std::tuple<Args...> args;
  Wrt(std::tuple<Args...> args) : args(args){};
};

template <typename... Args> struct ParallelWrt {
  std::tuple<Args...> args;
  ParallelWrt(std::tuple<Args...> args) : args(args){};
};

template <typename... Elem> constexpr auto seed(std::tuple<Elem...> wrt) {
  std::apply([](auto &&...args) { ((args.grad = 1), ...); }, wrt);
}

template <typename... Elem> constexpr auto unseed(std::tuple<Elem...> wrt) {
  std::apply([](auto &&...args) { ((args.grad = 0), ...); }, wrt);
}

template <typename... Args> constexpr auto at(Args &&...args) {
  return At<Args...>(std::forward_as_tuple<Args...>(args...));
}

template <typename... Args> constexpr auto wrt(Args &&...args) {
  return Wrt<Args...>(std::forward_as_tuple<Args...>(args...));
}

template <typename... Args> constexpr auto parallelWrt(Args &&...args) {
  return ParallelWrt<Args...>(std::forward_as_tuple<Args...>(args...));
}

namespace detail {
template <size_t I = 0, size_t W = 1, typename T, typename... Args>
auto vectorize_scalar_argument(std::tuple<Args...> &args,
                               std::array<T, W> &vectorized_args) {
  const auto current_arg = std::get<I>(args);
  using grad_t = decltype(current_arg.grad);

  auto vectorized_grad = generic_vec::vector<grad_t, W>();
  if (current_arg.grad == 1.0)
    vectorized_grad[I] = 1.0;

  const auto new_arg = var(current_arg.value, vectorized_grad);
  vectorized_args[I] = new_arg;

  if constexpr (I + 1 != sizeof...(Args))
    vectorize_scalar_argument<I + 1, W, T, Args...>(args, vectorized_args);
}

template <typename... Args> auto vectorize_args(std::tuple<Args...> args) {
  constexpr size_t arg_length = sizeof...(Args);

  const auto elem0 = std::get<0>(args);
  using val_t = decltype(elem0.value);
  using grad_t = decltype(elem0.grad);

  std::array<var<val_t, vector<grad_t, arg_length>>, arg_length>
      vectorized_args;

  vectorize_scalar_argument(args, vectorized_args);

  return vectorized_args;
}
} // namespace detail

template <typename F, typename... DVars, typename... Args>
auto eval(const F &&func, Wrt<DVars...> wrt, At<Args...> at) {
  seed(wrt.args);

  const auto ans = std::apply(func, at.args);

  unseed(wrt.args);

  return ans;
}

template <typename F, typename... DVars, typename... Args>
auto eval(const F &&func, ParallelWrt<DVars...> wrt, At<Args...> at) {
  seed(wrt.args);

  const auto ans = std::apply(func, detail::vectorize_args(at.args));

  unseed(wrt.args);

  return ans;
}

} // End namespace autodiff

using autodiff::at;
using autodiff::eval;
using autodiff::parallelWrt;
using autodiff::var;
using autodiff::wrt;

} // End namespace wfwdiff

#endif // WFWDIFF_AUTODIFF_H
