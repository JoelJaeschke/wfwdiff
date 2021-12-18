#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <wfwdiff.hpp>
#include <cmath>
#include <iostream>

using val_t = double;
using scalar_t = wfwdiff::var<val_t, val_t>;
using vec_t = wfwdiff::var<val_t, wfwdiff::vector<val_t, 4>>;

vec_t f(vec_t x, vec_t y) {
    return std::exp(x) * std::cos(y);
}

TEST_CASE("Test vectorized autodiff", "[core]") {
    scalar_t x = 3.2;
    scalar_t y = 2.1;

    const auto ans =
        wfwdiff::eval(f, wfwdiff::parallelWrt(x, y), wfwdiff::at(x, y));

    REQUIRE(ans.value == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans.grad[0] == Approx(-12.385).epsilon(0.001));
    REQUIRE(ans.grad[1] == Approx(-21.176).epsilon(0.001));
}