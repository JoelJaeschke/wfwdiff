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

    auto operator+=(const T& rhs) = delete;
    auto operator-=(const T& rhs) = delete;
    auto operator*=(const T& rhs) = delete;
    auto operator/=(const T& rhs) = delete;
};

template <>
struct vector<double, 4> {
    __m256d storage;

    vector(double x0, double x1, double x2, double x3) {
        storage = _mm256_set_pd(x3, x2, x1, x0);
    }

    vector(const __m256d data) : storage(data){};

    template <size_t width>
    vector(std::array<double, width> elems) {
        static_assert(elems.size() >= 4,
                      "Need at least 4 doubles to saturate register");
        storage = _mm256_set_pd(elems[3], elems[2], elems[1], elems[0]);
    }

    ~vector() = default;

    const double operator[](const size_t idx) { return storage[idx]; }

    vector<double, 4> operator+(const vector<double, 4>& rhs) {
        return vector(_mm256_add_pd(storage, rhs.storage));
    };

    vector<double, 4> operator-(const vector<double, 4>& rhs) {
        return vector(_mm256_sub_pd(storage, rhs.storage));
    };
    vector<double, 4> operator*(const vector<double, 4>& rhs) {
        return vector(_mm256_mul_pd(storage, rhs.storage));
    };

    vector<double, 4> operator/(const vector<double, 4>& rhs) {
        return vector(_mm256_div_pd(storage, rhs.storage));
    };

    vector<double, 4>& operator+=(const vector<double, 4>& rhs) {
        storage = _mm256_add_pd(storage, rhs.storage);
        return *this;
    };

    vector<double, 4>& operator-=(const vector<double, 4>& rhs) {
        storage = _mm256_sub_pd(storage, rhs.storage);
        return *this;
    };

    vector<double, 4>& operator*=(const vector<double, 4>& rhs) {
        storage = _mm256_mul_pd(storage, rhs.storage);
        return *this;
    };

    vector<double, 4>& operator/=(const vector<double, 4>& rhs) {
        storage = _mm256_div_pd(storage, rhs.storage);
        return *this;
    };
};

std::ostream& operator<<(std::ostream& os, vector<double, 4> vec) {
    std::cout << "{" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", "
              << vec[3] << "}";
    return os;
};

}  // namespace generic_vec

using generic_vec::vector;
}  // namespace wfwdiff

#endif
