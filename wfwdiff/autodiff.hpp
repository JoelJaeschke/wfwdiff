#ifndef WFWDIFF_AUTODIFF_H
#define WFWDIFF_AUTODIFF_H

#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>

namespace wfwdiff {
namespace autodiff {

template <typename T>
struct var {
    T value;
    T grad;

    constexpr var(T value) : value(value), grad(0){};
    constexpr var(T value, T grad) : value(value), grad(grad){};

    constexpr const var<T> operator+(const var<T> rhs) const {
        return var<T>(value + rhs.value, grad + rhs.grad);
    };

    constexpr const var<T> operator-(const var<T> rhs) const {
        return var<T>(value - rhs.value, grad - rhs.grad);
    };

    constexpr const var<T> operator*(const var<T> rhs) const {
        return var<T>(value * rhs.value, value * rhs.grad + rhs.value * grad);
    };

    constexpr const var<T> operator/(const var<T> rhs) const {
        return var<T>(
            value / rhs.value,
            grad / rhs.value - rhs.grad * value / std::pow(rhs.val, 2));
    };

    constexpr const var<T> operator+=(const var<T> rhs) const {
        return this + rhs;
    };
    constexpr const var<T> operator-=(const var<T> rhs) const {
        return this - rhs;
    };
    constexpr const var<T> operator*=(const var<T> rhs) const {
        return this * rhs;
    };
    constexpr const var<T> operator/=(const var<T> rhs) const {
        return this / rhs;
    };
};

template <typename... Elem>
constexpr auto seed(std::tuple<Elem...> elements) {
    std::apply([](auto&&... args) { ((args.grad = 1), ...); }, elements);
}

template <typename... Elem>
constexpr auto unseed(std::tuple<Elem...> elements) {
    std::apply([](auto&&... args) { ((args.grad = 0), ...); }, elements);
}

template <typename... Args>
constexpr auto at(Args&&... args) {
    return std::forward_as_tuple<Args...>(args...);
}

template <typename... Args>
constexpr auto wrt(Args&&... args) {
    return std::forward_as_tuple<Args...>(args...);
}

template <typename F, typename... Wrt, typename... Args>
const auto eval(const F&& func, std::tuple<Wrt...> wrt,
                std::tuple<Args...> args) {
    seed(wrt);

    const auto ans = std::apply(func, args);

    unseed(wrt);

    return ans;
};

}  // End namespace autodiff

using autodiff::at;
using autodiff::eval;
using autodiff::var;
using autodiff::wrt;

}  // End namespace wfwdiff

#endif
