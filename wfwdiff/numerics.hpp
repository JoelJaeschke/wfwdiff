#ifndef NUMERICS_H
#define NUMERICS_H

#include <cmath>

#include "autodiff.hpp"

using wfwdiff::var;

namespace std {

template <typename T>
const var<T> sin(const var<T> x) {
    return var<T>(sin(x.value), x.grad * cos(x.value));
};

template <typename T>
const var<T> cos(const var<T> x) {
    return var<T>(cos(x.value), -x.grad * sin(x.value));
};
template <typename T>
const var<T> tan(const var<T> x) {
    return var<T>(tan(x.value), x.grad * pow(sec(x.value), 2));
};
template <typename T>
const var<T> acos(const var<T> x) {
    return var<T>(acos(x.value), -x.grad / sqrt(1 - pow(x.value, 2)));
};
template <typename T>
const var<T> asin(const var<T> x) {
    return var<T>(asin(x.value), x.grad / sqrt(1 - pow(x.value, 2)));
};
template <typename T>
const var<T> atan(const var<T> x) {
    return var<T>(atan(x.value), x.grad / sqrt(1 + pow(x.value, 2)));
};

template <typename T>
const var<T> exp(const var<T> x) {
    return var<T>(exp(x.value), x.grad * exp(x.value));
};

};  // namespace std

#endif
