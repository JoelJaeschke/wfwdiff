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
using vec_t = wfwdiff::var<double, wfwdiff::vector<double, 3>>;

// f is the function we will try to minimize
vec_t f(vec_t x, vec_t y, vec_t z) {
  return x * x + y * y + 0.4 * std::cos(3. * x) - 0.5 * std::sin(3. * y) +
         0.5 * x + 2.3 * std::cos(2. * z);
}

int main() {
  // We initialize x and y to our starting point
  scalar_t x = 3.4;
  scalar_t y = 1.3;
  scalar_t z = 6.43;

  std::cout << "Starting search at (x=" << x.value << ",y=" << y.value
            << ")!\n";

  const double gamma = 0.005; // gamma will be our learning rate
  double dx = 1000, dy = 1000,
         dz = 1000;                 // step difference, for loop termination
  const double lower_bound = 1e-13; // Loop termination bound

  while ((dx + dy + dz) > lower_bound) {
    const auto ans =
        wfwdiff::eval(f, wfwdiff::parallelWrt(x, y, z), wfwdiff::at(x, y, z));

    dx = x.value;
    dy = y.value; // dx and dy temporarily store old x and y
    dz = z.value;

    x = x - gamma * ans.grad[0];
    y = y - gamma * ans.grad[1];
    z = z - gamma * ans.grad[2];

    dx = std::abs(x.value - dx);
    dy = std::abs(y.value - dy);
    dz = std::abs(z.value - dz);
  }

  std::cout << "Found (local) minimum at (x=" << x.value << ",y=" << y.value
            << ",z=" << z.value << ")!\n";

  return 0;
}