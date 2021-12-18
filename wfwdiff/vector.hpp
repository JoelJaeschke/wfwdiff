#ifndef WFWDIFF_VECTOR_H
#define WFWDIFF_VECTOR_H

#include <immintrin.h>

#include <array>
#include <cassert>
#include <iostream>

namespace wfwdiff {
namespace generic_vec {

template <typename T, size_t width>
struct vector {
    T storage;
    vector(T initializer) : storage(initializer){};
    ~vector() = default;

    auto operator+(const T& rhs) = delete;
    auto operator-(const T& rhs) = delete;
    auto operator*(const T& rhs) = delete;
    auto operator/(const T& rhs) = delete;

    auto operator=(const T& rhs) = delete;
    auto operator+=(const T& rhs) = delete;
    auto operator-=(const T& rhs) = delete;
    auto operator*=(const T& rhs) = delete;
    auto operator/=(const T& rhs) = delete;
};

template <>
struct vector<double, 4> {
    using val_t = vector<double, 4>;
    __m256d storage;

    constexpr vector(): storage{0.0,0.0,0.0,0.0} {}

    constexpr vector(double x0, double x1, double x2, double x3):storage{x0,x1,x2,x3} {}

    constexpr vector(const __m256d data) : storage(data){};

    template <size_t width>
    vector(std::array<double, width> elems) {
        static_assert(elems.size() >= 4,
                      "Need at least 4 doubles to saturate register");
        storage = {elems[3], elems[2], elems[1], elems[0]};
    }

    ~vector() = default;

    vector<double, 4>& operator=(const double num) {
        storage = _mm256_set1_pd(num);
        return *this;
    };
    
    vector<double, 4>& operator=(const std::array<double, 4> numbers) {
        storage = _mm256_set_pd(numbers[3], numbers[2], numbers[1], numbers[0]);
        return *this;
    };
    
    auto operator[](const size_t idx) const { return storage[idx]; }

    val_t operator+(const val_t& rhs) const {
        return _mm256_add_pd(storage, rhs.storage);
    };

    val_t operator+(const double rhs) const {
        return _mm256_add_pd(_mm256_set1_pd(rhs), storage);
    };

    val_t operator-(const val_t& rhs) const {
        return _mm256_sub_pd(storage, rhs.storage);
    };

    val_t operator-(const double rhs) const {
        return _mm256_sub_pd(storage, _mm256_set1_pd(rhs));
    };

    val_t operator*(const val_t& rhs) const {
        return _mm256_mul_pd(storage, rhs.storage);
    };

    val_t operator*(const double rhs) const {
        return _mm256_mul_pd(storage, _mm256_set1_pd(rhs));
    };

    val_t operator/(const val_t& rhs) const {
        return _mm256_div_pd(storage, rhs.storage);
    };

    val_t operator/(const double rhs) const {
        return _mm256_div_pd(storage, _mm256_set1_pd(rhs));
    };

    val_t& operator+=(const val_t& rhs) {
        storage = _mm256_add_pd(storage, rhs.storage);
        return *this;
    };

    val_t& operator-=(const val_t& rhs) {
        storage = _mm256_sub_pd(storage, rhs.storage);
        return *this;
    };

    val_t& operator*=(const val_t& rhs) {
        storage = _mm256_mul_pd(storage, rhs.storage);
        return *this;
    };

    val_t& operator/=(const val_t& rhs) {
        storage = _mm256_div_pd(storage, rhs.storage);
        return *this;
    };

    auto set(const size_t index, double value) {
        storage[index] = value;
    }

    friend std::ostream& operator<<(std::ostream& os, const val_t& vec) {
        std::cout << "{" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", "
                  << vec[3] << "}";
        return os;
    };
};
}  // namespace generic_vec

using generic_vec::vector;
}  // namespace wfwdiff

#endif
