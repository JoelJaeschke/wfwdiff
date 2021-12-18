#ifndef WFWDIFF_AUTODIFF_H
#define WFWDIFF_AUTODIFF_H

#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>

#include "vector.hpp"

namespace wfwdiff {
namespace autodiff {

template <typename T, typename U>
struct var {
    T value;
    U grad;

    constexpr var(): value(), grad(){};
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
        return var<T, U>(
            value / rhs.value,
            grad / rhs.value - rhs.grad * value / std::pow(rhs.val, 2));
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
};

template <typename... Args>
struct At {
    std::tuple<Args...> args;
    At(std::tuple<Args...> args) : args(args){};
};

template <typename... Args>
struct Wrt {
    std::tuple<Args...> args;
    Wrt(std::tuple<Args...> args) : args(args){};
};

template <typename... Args>
struct ParallelWrt {
    std::tuple<Args...> args;
    ParallelWrt(std::tuple<Args...> args) : args(args){};
};

template <typename... Elem>
constexpr auto seed(std::tuple<Elem...> wrt) {
    std::apply([](auto&&... args) { ((args.grad = 1), ...); }, wrt);
}

template <typename... Elem>
constexpr auto unseed(std::tuple<Elem...> wrt) {
    std::apply([](auto&&... args) { ((args.grad = 0), ...); }, wrt);
}

template <typename... Args>
constexpr auto at(Args&&... args) {
    return At<Args...>(std::forward_as_tuple<Args...>(args...));
}

template <typename... Args>
constexpr auto wrt(Args&&... args) {
    return Wrt<Args...>(std::forward_as_tuple<Args...>(args...));
}

template <typename... Args>
constexpr auto parallelWrt(Args&&... args) {
    return ParallelWrt<Args...>(std::forward_as_tuple<Args...>(args...));
}

template <size_t I = 0, size_t W = 1, typename T, typename... Args>
auto vectorize_scalar_argument(
    std::tuple<Args...>& args,
    std::array<T, W>& vectorized_args)
{
    const auto current_arg = std::get<I>(args);
    auto vectorized_grad = wfwdiff::vector<double, 4>();
    if (current_arg.grad == 1.0)
        vectorized_grad.set(I, 1.0);

    const auto new_arg = var<double, wfwdiff::vector<double, 4>>(
        current_arg.value,
        vectorized_grad);
    vectorized_args[I] = new_arg;

    if constexpr(I+1 != sizeof...(Args))
        vectorize_scalar_argument<I+1, W, T, Args...>(args, vectorized_args);
}

template <typename...> struct WhichType;

template <typename... Args>
auto vectorize_args(std::tuple<Args...> args) {
    constexpr size_t arg_length = sizeof...(Args);
    std::array<var<double, wfwdiff::vector<double, 4>>,
                    arg_length> vectorized_args;

    vectorize_scalar_argument(args, vectorized_args);

    return vectorized_args;
}

template <typename F, typename... DVars, typename... Args>
auto eval(const F&& func, Wrt<DVars...> wrt, At<Args...> at) {
    seed(wrt.args);

    const auto ans = std::apply(func, at.args);

    unseed(wrt.args);

    return ans;
}

// Instead of
template <typename F, typename... DVars, typename... Args>
auto eval(const F&& func, ParallelWrt<DVars...> wrt, At<Args...> at) {
    seed(wrt.args);

    auto converted_args = vectorize_args(at.args);

    const auto ans = std::apply(func, converted_args);

    unseed(wrt.args);

    return ans;
}

}  // End namespace autodiff

using autodiff::at;
using autodiff::eval;
using autodiff::var;
using autodiff::wrt;
using autodiff::parallelWrt;

}  // End namespace wfwdiff

#endif
