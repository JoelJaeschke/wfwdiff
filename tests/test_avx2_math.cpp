#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include <wfwdiff/wfwdiff.hpp>

using vec_t = wfwdiff::vector<double, 4>;

TEST_CASE("Test exponential avx2", "[core, math]") {
    vec_t vec = wfwdiff::vector<double, 4>(1.0,2.0,3.0,4.0);
    vec_t exp_vec = vec.exp();

    std::cout << exp_vec;
    
    vec_t exp_ref;
    for (std::size_t i = 0; i < 4; i++) {
        exp_ref[i] = std::cos(vec[i]);
    }
    
    REQUIRE(exp_ref[0] == Approx(exp_vec[0]).epsilon(0.00000000001));
    REQUIRE(exp_ref[1] == Approx(exp_vec[1]).epsilon(0.00000000001));
    REQUIRE(exp_ref[2] == Approx(exp_vec[2]).epsilon(0.00000000001));
    REQUIRE(exp_ref[3] == Approx(exp_vec[3]).epsilon(0.00000000001));
}

TEST_CASE("Test abs avx2", "[core, math]") {
    vec_t vec = wfwdiff::vector<double, 4>(-1.0,2.0,-3.0,-4.0);
    vec_t vec_ref = wfwdiff::vector<double, 4>(1.0,2.0,3.0,4.0);

    vec_t vec_abs = vec.abs();

    REQUIRE(vec_abs[0] == Approx(vec_ref[0]).epsilon(0.00000000001));
    REQUIRE(vec_abs[1] == Approx(vec_ref[1]).epsilon(0.00000000001));
    REQUIRE(vec_abs[2] == Approx(vec_ref[2]).epsilon(0.00000000001));
    REQUIRE(vec_abs[3] == Approx(vec_ref[3]).epsilon(0.00000000001));
}

TEST_CASE("Test pow2 avx2", "[core, math]") {
    vec_t vec = wfwdiff::vector<double, 4>(1.0,2.0,3.0,4.0);
    vec_t vec_pow = vec.pow2();


    vec_t vec_ref;
    for (std::size_t i = 0; i < 4; i++) {
        vec_ref[i] = std::pow(vec[i], 2);
    }

    REQUIRE(vec_ref[0] == Approx(vec_pow[0]).epsilon(0.00000000001));
    REQUIRE(vec_ref[1] == Approx(vec_pow[1]).epsilon(0.00000000001));
    REQUIRE(vec_ref[2] == Approx(vec_pow[2]).epsilon(0.00000000001));
    REQUIRE(vec_ref[3] == Approx(vec_pow[3]).epsilon(0.00000000001));
}

TEST_CASE("Test inv avx2", "[core, math]") {
    vec_t vec = wfwdiff::vector<double, 4>(1.0,2.0,3.0,4.0);
    vec_t vec_inv = vec.inv();

    vec_t vec_ref;
    for (std::size_t i = 0; i < 4; i++) {
        vec_ref[i] = 1 / vec[i];
    }

    REQUIRE(vec_ref[0] == Approx(vec_inv[0]).epsilon(0.00000000001));
    REQUIRE(vec_ref[1] == Approx(vec_inv[1]).epsilon(0.00000000001));
    REQUIRE(vec_ref[2] == Approx(vec_inv[2]).epsilon(0.00000000001));
    REQUIRE(vec_ref[3] == Approx(vec_inv[3]).epsilon(0.00000000001));
}