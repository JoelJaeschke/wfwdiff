#ifndef WFWDIFF_VECTOR_AVX2_H
#define WFWDIFF_VECTOR_AVX2_H

#include <immintrin.h>

#include <array>
#include <cassert>
#include <iostream>
#include <math.h>

#include "vector_base.hpp"

namespace wfwdiff {
namespace generic_vec {

template <>
struct vector<double, 4> {
    using val_t = vector<double, 4>;

   private:
    __m256d storage;

   public:
    constexpr vector() : storage{0.0, 0.0, 0.0, 0.0} {}

    constexpr vector(double x0, double x1, double x2, double x3)
        : storage{x0, x1, x2, x3} {}

    constexpr vector(const __m256d data) : storage(data){};

    constexpr vector(double val): storage{ val, val, val, val} {};

    template <size_t width>
    vector(const std::array<double, width>& elems) {
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

    const double operator[](const size_t idx) const { return storage[idx]; }

    double& operator[](const size_t idx) { return storage[idx]; }

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

    val_t operator>(const val_t& rhs) {
        return _mm256_cmp_pd(storage, rhs.storage, _CMP_LT_OQ);
    }

    val_t operator>(const double rhs) {
        auto rhs_storage = _mm256_set1_pd(rhs);
        return _mm256_cmp_pd(storage, rhs_storage, _CMP_LT_OQ);
    }

    val_t operator<(const val_t& rhs) {
        return _mm256_cmp_pd(storage, rhs.storage, _CMP_GT_OQ);
    }

    val_t operator<(const double rhs) {
        auto rhs_storage = _mm256_set1_pd(rhs);
        return _mm256_cmp_pd(storage, rhs_storage, _CMP_GT_OQ);
    }

    val_t operator==(const val_t& rhs) {
        return _mm256_cmp_pd(storage, rhs.storage, _CMP_EQ_OQ);
    }

    val_t operator==(const double rhs) {
        auto rhs_storage = _mm256_set1_pd(rhs);
        return _mm256_cmp_pd(storage, rhs_storage, _CMP_EQ_OQ);
    }

    val_t operator<=(const val_t& rhs) {
        return _mm256_cmp_pd(storage, rhs.storage, _CMP_LE_OQ);
    }

    val_t operator<=(const double rhs) {
        auto rhs_storage = _mm256_set1_pd(rhs);
        return _mm256_cmp_pd(storage, rhs_storage, _CMP_LE_OQ);
    }

    val_t operator>=(const val_t& rhs) {
        return _mm256_cmp_pd(storage, rhs.storage, _CMP_GE_OQ);
    }

    val_t operator>=(const double rhs) {
        auto rhs_storage = _mm256_set1_pd(rhs);
        return _mm256_cmp_pd(storage, rhs_storage, _CMP_GE_OQ);
    }

    bool all_true(const __m256d mask) const {
        return 0xF == _mm256_movemask_pd(mask);
    }

    val_t masked_set(const double val, const val_t mask) {
        const auto vals = _mm256_set1_pd(val);

        return _mm256_blendv_pd(vals, storage, mask.storage);
    }

    val_t masked_set(const val_t vals, const val_t mask) {
        return _mm256_blendv_pd(vals.storage, storage, mask.storage);
    }

    val_t& masked_set_inplace(const double val, const val_t mask) {
        auto ones = _mm256_set1_pd(val);
        storage = _mm256_blendv_pd(ones, storage, mask.storage);

        return *this;
    }

    val_t& masked_set_inplace(const val_t vals, const val_t mask) {
        storage = _mm256_blendv_pd(vals.storage, storage, mask.storage);

        return *this;
    }

    val_t pow2() {
        return _mm256_mul_pd(storage, storage);
    }

    val_t inv() {
        return _mm256_div_pd(_mm256_set1_pd(1.0), storage);
    }

    val_t abs() const {
        static const auto sign_mask = _mm256_set1_pd(-0.);
        return _mm256_andnot_pd(sign_mask, storage);
    }

    val_t exp() const {
        static const val_t M_LN2_VEC = _mm256_set1_pd(M_LN2);
        static const std::array<val_t, 14> coeffs = {
            1.000000000000000,
            1.000000000000000,
            0.500000000000002,
            0.166666666666680,
            0.041666666666727,
            0.008333333333342,
            0.001388888888388,
            1.984126978734782e-4,
            2.480158866546844e-5,
            2.755734045527853e-6,
            2.755715675968011e-7,
            2.504861486483735e-8,
            2.088459690899721e-9,
            1.632461784798319e-10
        };

        // https://www.pseudorandom.com/implementing-exp#section-24
        assert(all_true(-709 <= storage) && all_true(storage <= 709));

        val_t x0 = this->abs();
        val_t x1 = (x0 - M_LN2_VEC) - 0.5;
        val_t k = _mm256_ceil_pd(x1.storage);
        val_t r = x0 - (k * M_LN2_VEC);
        val_t pn = 1.143364767943110e-11;

        for (auto coeff = coeffs.rbegin(); coeff != coeffs.rend(); coeff++) {
            pn = pn * r + *coeff;
        }

        pn *= k.pow2();

        // Set all values where input is zero to one
//        const val_t zero_mask = storage == 0.0;
//        pn.masked_set_inplace(1.0, zero_mask);

//        const val_t inv_mask = storage < 0;
//        const val_t inverted = pn.inv();

//        pn.masked_set_inplace(inverted, inv_mask);

        return pn;
    }

    friend std::ostream& operator<<(std::ostream& os, const val_t& vec) {
        std::cout << "{" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", "
                  << vec[3] << "}";
        return os;
    };
};

}  // namespace generic_vec
}  // namespace wfwdiff

#endif  // WFWDIFF_VECTOR_AVX2_H
