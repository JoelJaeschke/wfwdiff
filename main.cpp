#include "wfwdiff/wfwdiff.hpp"

int main() {
    wfwdiff::vector<double, 4> x(0.4, 0.1, 0.3, 0.8);
    wfwdiff::vector<double, 4> y(1., 4., 3., 2.);

    auto c = x + y;

    std::cout << c << "\n";
    return 0;
}
