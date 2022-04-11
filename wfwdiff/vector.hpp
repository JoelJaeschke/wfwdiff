#ifndef WFWDIFF_VECTOR_BASE_H
#define WFWDIFF_VECTOR_BASE_H

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>

// clang-format off
#include "xsimd/xsimd.hpp"
#include "xsimd/stl/algorithms.hpp"
// clang-format on

namespace wfwdiff {
namespace generic_vec {

static constexpr bool SIMD_ACTIVE = false;
template <typename T, size_t width, bool USE_SIMD = SIMD_ACTIVE> struct vector;

template <typename T, size_t width> struct vector<T, width, false> {
private:
  std::array<T, width> storage_;

public:
  vector() : storage_(){};

  template <typename... Ts> vector(Ts... vals) : storage_({vals...}) {}

  vector(const std::array<T, width> &input) : storage_(input){};
  vector(const T initializer) {
    std::fill_n(storage_.begin(), width, initializer);
  };
  vector(const vector<T, width> &vec) : storage_(vec.data()){};

  ~vector() = default;

  auto operator+(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   result.begin(), std::plus<T>());

    return vector(result);
  };

  auto operator-(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   result.begin(), std::minus<T>());

    return vector(result);
  };

  auto operator*(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   result.begin(), std::multiplies<T>());

    return vector(result);
  };

  auto operator/(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   result.begin(), std::divides<T>());

    return vector(result);
  };

  vector<T, width> &operator+=(const vector<T, width> &rhs) {
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   storage_.begin(), std::plus<T>());

    return *this;
  };

  vector<T, width> &operator-=(const vector<T, width> &rhs) {
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   storage_.begin(), std::minus<T>());

    return *this;
  };

  vector<T, width> &operator*=(const vector<T, width> &rhs) {
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   storage_.begin(), std::multiplies<T>());

    return *this;
  };

  vector<T, width> &operator/=(const vector<T, width> &rhs) {
    std::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                   storage_.begin(), std::divides<T>());

    return *this;
  };

  const T &operator[](const size_t idx) const { return storage_[idx]; }

  T &operator[](const size_t idx) { return storage_[idx]; }

  const std::array<T, width> &data() const { return storage_; };

  std::array<T, width> &data() { return storage_; };
};

template <typename T, size_t width> struct vector<T, width, true> {
private:
  // TODO: Check whether alignas is necessary
  alignas(16) std::array<T, width> storage_;

public:
  vector() : storage_(){};

  template <typename... Ts> vector(Ts... vals) : storage_({vals...}) {}

  vector(const std::array<T, width> &input) : storage_(input){};
  vector(const T initializer) {
    std::fill_n(storage_.begin(), width, initializer);
  };
  vector(const vector<T, width> &vec) : storage_(vec.data()){};

  ~vector() = default;

  auto operator+(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     result.begin(),
                     [](const auto &x, const auto &y) { return x + y; });

    return vector(result);
  };

  auto operator-(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     result.begin(),
                     [](const auto &x, const auto &y) { return x - y; });

    return vector(result);
  };

  auto operator*(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     result.begin(),
                     [](const auto &x, const auto &y) { return x * y; });

    return vector(result);
  };

  auto operator/(const vector<T, width> &rhs) const {
    std::array<T, width> result{};
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     result.begin(),
                     [](const auto &x, const auto &y) { return x / y; });

    return vector(result);
  };

  vector<T, width> &operator+=(const vector<T, width> &rhs) {
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     storage_.begin(),
                     [](const auto &x, const auto &y) { x + y; });

    return *this;
  };

  vector<T, width> &operator-=(const vector<T, width> &rhs) {
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     storage_.begin(),
                     [](const auto &x, const auto &y) { x - y; });

    return *this;
  };

  vector<T, width> &operator*=(const vector<T, width> &rhs) {
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     storage_.begin(),
                     [](const auto &x, const auto &y) { x *y; });

    return *this;
  };

  vector<T, width> &operator/=(const vector<T, width> &rhs) {
    xsimd::transform(storage_.begin(), storage_.end(), rhs.data().begin(),
                     storage_.begin(),
                     [](const auto &x, const auto &y) { x / y; });

    return *this;
  };

  const T &operator[](const size_t idx) const { return storage_[idx]; }

  T &operator[](const size_t idx) { return storage_[idx]; }

  const std::array<T, width> &data() const { return storage_; };

  std::array<T, width> &data() { return storage_; };
};

template <typename T, size_t width, bool USE_SIMD>
std::ostream &operator<<(std::ostream &os,
                         const vector<T, width, USE_SIMD> vec) {
  os << "[";

  std::for_each(vec.data().begin(), std::prev(vec.data().end()),
                [&os](const auto &elem) { os << elem << ","; });

  os << vec.data().back() << "]";

  return os;
}

namespace detail {
template <typename T>
concept is_arithmetic_vector = requires(T a, T b) {
  a + b;
  a - b;
  a / b;
  a *b;
};

template <typename T> struct is_differentiable {
  using val_t = std::decay_t<decltype(std::declval<T &>()[0])>;
  static constexpr bool value{std::integral<val_t> ||
                              std::floating_point<val_t>};
};

template <typename T>
inline constexpr bool is_differentiable_v = is_differentiable<T>::value;
} // namespace detail

template <typename T>
concept differentiable_vector =
    detail::is_arithmetic_vector<T> && detail::is_differentiable_v<T>;

} // namespace generic_vec

using generic_vec::vector;

} // namespace wfwdiff

#endif // WFWDIFF_VECTOR_BASE_H