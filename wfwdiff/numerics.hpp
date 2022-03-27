#ifndef WFWDIFF_NUMERICS_H
#define WFWDIFF_NUMERICS_H

#include <cmath>

#include "autodiff.hpp"
#include "vector.hpp"

namespace std {
using wfwdiff::var;

template <typename T, typename U>
var<T, U> sin(const var<T, U> x) {
    return var<T, U>(sin(x.value), x.grad * cos(x.value));
}
template <typename T, typename U>
var<T, U> cos(const var<T, U> x) {
    return var<T, U>(cos(x.value), x.grad * -sin(x.value));
}
template <typename T, typename U>
var<T, U> tan(const var<T, U> x) {
    return var<T, U>(tan(x.value), x.grad * pow(sec(x.value), 2));
}
template <typename T, typename U>
var<T, U> acos(const var<T, U> x) {
    return var<T, U>(acos(x.value), -x.grad / sqrt(1 - pow(x.value, 2)));
}
template <typename T, typename U>
var<T, U> asin(const var<T, U> x) {
    return var<T, U>(asin(x.value), x.grad / sqrt(1 - pow(x.value, 2)));
}
template <typename T, typename U>
var<T, U> atan(const var<T, U> x) {
    return var<T, U>(atan(x.value), x.grad / sqrt(1 + pow(x.value, 2)));
}
template <typename T, typename U>
var<T, U> exp(const var<T, U> x) {
    return var<T, U>(exp(x.value), x.grad * exp(x.value));
}
template <typename T, typename U>
var<T, U> log(const var<T, U> x) {
    return var<T, U>(log(x.value), x.grad * 1/x.value);
}
}// namespace std

#endif // WFWDIFF_NUMERICS_H
