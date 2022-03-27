#ifndef NUMERICS_H
#define NUMERICS_H

#include <cmath>
#include <immintrin.h>

#include "autodiff.hpp"
#include "vector/vector.hpp"

namespace std {
using wfwdiff::var;
using vec_t = wfwdiff::vector<double, 4>;

vec_t sin(const vec_t x) {
    
    return x;
}

vec_t cos(const vec_t in) {
    vec_t out;
    for (std::size_t i = 0; i < 4; i++) {
        out[i] = std::cos(in[i]);
    }
    return out;
}

vec_t tan(const vec_t x) {
    return x;
}

vec_t acos(const vec_t x) {
    return x;
}

vec_t asin(const vec_t x) {
    return x;
}

vec_t atan(const vec_t x) {
    return x;
}

#if 0
vec_t exp(const vec_t in) {
    vec_t out;
    for (std::size_t i = 0; i < 4; i++) {
        out[i] = std::exp(in[i]);
    }
    return out;
}
#else
vec_t exp(const vec_t in) {
    return in.exp();
}
#endif

vec_t log(const vec_t x) {
    return x;
}
} // namespace std

namespace std {
template <typename T, typename U>
var<T, U> sin(const var<T, U> x) {
    return var<T, U>(sin(x.value), x.grad * cos(x.value));
}
template <typename T, typename U>
var<T, U> cos(const var<T, U> x) {
    return var<T, U>(cos(x.value), x.grad * -1 * sin(x.value));
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

#endif
