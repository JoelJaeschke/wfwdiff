//
// Created by joel on 3/27/22.
//
// An example demonstrating how the autodiff library can be used to
// run gradient descent without requiring explicit numerical computation
// of the gradient

#include <cmath>
#include <wfwdiff/wfwdiff.hpp>

// Some convenience definitions to save typing
// We will use doubles for values and gradients.
using scalar_t = wfwdiff::var<double, double>;
using vec_t = wfwdiff::var<double, wfwdiff::vector<double, 2>>;

// f is the function we will try to minimize
vec_t f(vec_t x, vec_t y) {
    return x*x + y*y + std::cos(x*3) * 0.4 - std::sin(y*3)*0.5 + x*0.5;
}

int main() {
    // We initialize x and y to our starting point
    scalar_t x = 3.4;
    scalar_t y = 1.3;

    std::cout << "Starting search at (x=" << x.value << ",y="
              << y.value << ")!\n";

    const double gamma = 0.3; // gamma will be our learning rate
    double dx = 1000, dy = 1000; // step difference, for loop termination
    const double lower_bound = 1e-5; // Loop termination bound

    while ((dx + dy) > lower_bound) {
        const auto ans =
            wfwdiff::eval(f, wfwdiff::parallelWrt(x,y), wfwdiff::at(x,y));

        dx = x.value; dy = y.value; // dx and dy temporarily store old x and y

        x = x - gamma * ans.grad[0];
        y = y - gamma * ans.grad[1];

        dx = std::abs(x.value - dx);
        dy = std::abs(y.value - dy);
    }

    std::cout << "Found (local) minimum at (x=" << x.value
              << ",y=" << y.value << ")!\n";

    return 0;
}