#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <wfwdiff.hpp>
#include <cmath>

using val_t = wfwdiff::var<double, double>;
val_t f(val_t x, val_t y) {
    return std::exp(x) * std::cos(y);
}

TEST_CASE("Test scalar autodiff", "[core]") {
    val_t x = 3.2;
    val_t y = 2.1;

    const auto ans_dx =
        wfwdiff::eval(f, wfwdiff::wrt(x), wfwdiff::at(x, y));

    const auto ans_dy =
        wfwdiff::eval(f, wfwdiff::wrt(y), wfwdiff::at(x, y));

    REQUIRE(ans_dx.value == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans_dy.value == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans_dx.grad == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans_dy.grad == Approx(-21.176).epsilon(0.001));
}