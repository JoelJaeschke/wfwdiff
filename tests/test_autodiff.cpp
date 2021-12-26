#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include <wfwdiff/wfwdiff.hpp>

using val_t = double;
using scalar_t = wfwdiff::var<val_t, val_t>;
using vec_t = wfwdiff::var<
    val_t, wfwdiff::vector<
               val_t, wfwdiff::generic_vec::fastest_vec_available<2>::size>>;

scalar_t f_scalar(scalar_t x, scalar_t y) { return std::exp(x) * std::cos(y); }

vec_t f_vec(vec_t x, vec_t y) { return std::exp(x) * std::cos(y); }

TEST_CASE("Test scalar autodiff", "[core]") {
    scalar_t x = 3.2;
    scalar_t y = 2.1;

    const auto ans_dx =
        wfwdiff::eval(f_scalar, wfwdiff::wrt(x), wfwdiff::at(x, y));

    const auto ans_dy =
        wfwdiff::eval(f_scalar, wfwdiff::wrt(y), wfwdiff::at(x, y));

    REQUIRE(ans_dx.value == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans_dy.value == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans_dx.grad == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans_dy.grad == Approx(-21.176).epsilon(0.001));
}

TEST_CASE("Test vectorized autodiff", "[core]") {
    scalar_t x = 3.2;
    scalar_t y = 2.1;

    const auto ans =
        wfwdiff::eval(f_vec, wfwdiff::parallelWrt(x, y), wfwdiff::at(x, y));

    REQUIRE(ans.value == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans.grad[0] == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans.grad[1] == Approx(-21.176).epsilon(0.001));
}