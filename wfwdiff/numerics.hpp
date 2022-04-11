#ifndef WFWDIFF_NUMERICS_H
#define WFWDIFF_NUMERICS_H

#include <cmath>

#include "autodiff.hpp"
#include "vector.hpp"

namespace std {

template <typename T, typename U>
wfwdiff::var<T, U> sin(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(sin(x.value), x.grad * cos(x.value));
}
template <typename T, typename U>
wfwdiff::var<T, U> cos(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(cos(x.value), x.grad * -sin(x.value));
}
template <typename T, typename U>
wfwdiff::var<T, U> tan(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(tan(x.value), x.grad * pow(sec(x.value), 2));
}
template <typename T, typename U>
wfwdiff::var<T, U> acos(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(acos(x.value), -x.grad / sqrt(1 - pow(x.value, 2)));
}
template <typename T, typename U>
wfwdiff::var<T, U> asin(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(asin(x.value), x.grad / sqrt(1 - pow(x.value, 2)));
}
template <typename T, typename U>
wfwdiff::var<T, U> atan(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(atan(x.value), x.grad / sqrt(1 + pow(x.value, 2)));
}
template <typename T, typename U>
wfwdiff::var<T, U> exp(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(exp(x.value), x.grad * exp(x.value));
}
template <typename T, typename U>
wfwdiff::var<T, U> log(const wfwdiff::var<T, U> x) {
  return wfwdiff::var<T, U>(log(x.value), x.grad * 1 / x.value);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
sin(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::sin);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
cos(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::cos);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
tan(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::tan);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
asin(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::asin);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
acos(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::acos);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
atan(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::atan);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
exp(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::exp);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false>
log(const wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::array<T, width> result = x.data();
  std::for_each(result.begin(), result.end(), std::log);

  return wfwdiff::generic_vec::vector<T, width, false>(result);
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
sin(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::sin);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
cos(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::cos);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
tan(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::tan);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
asin(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::asin);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
acos(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::acos);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
atan(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::atan);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
exp(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::exp);

  return x;
}

template <typename T, size_t width>
wfwdiff::generic_vec::vector<T, width, false> &
log(wfwdiff::generic_vec::vector<T, width, false> &x) {
  std::for_each(x.data().begin(), x.data().end(), std::log);

  return x;
}

} // namespace std

template <typename T, typename U>
requires wfwdiff::autodiff::detail::Differentiable<T> &&
    wfwdiff::autodiff::detail::Differentiable<U>
        wfwdiff::var<T, U>
operator+(T scalar, wfwdiff::var<T, U> var) { return var + scalar; }

template <typename T, typename U>
requires wfwdiff::autodiff::detail::Differentiable<T> &&
    wfwdiff::autodiff::detail::Differentiable<U>
        wfwdiff::var<T, U>
operator-(T scalar, wfwdiff::var<T, U> var) { return scalar - var; }

template <typename T, typename U>
requires wfwdiff::autodiff::detail::Differentiable<T> &&
    wfwdiff::autodiff::detail::Differentiable<U>
        wfwdiff::var<T, U>
operator*(T scalar, wfwdiff::var<T, U> var) { return var * scalar; }

template <typename T, typename U>
requires wfwdiff::autodiff::detail::Differentiable<T> &&
    wfwdiff::autodiff::detail::Differentiable<U>
        wfwdiff::var<T, U>
operator/(T scalar, wfwdiff::var<T, U> var) { return scalar / var; }

#endif // WFWDIFF_NUMERICS_H
