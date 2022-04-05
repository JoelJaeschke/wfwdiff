# WFWDiff
WFWDiff is a header-only library that can be used to automatically differentiate
mathematical functions. It is an alternative approach to numerical differentiation
(for example using finite differences) that does not suffer from truncation errors.

WFWDiff implements only forward-mode autodiff at the moment. This means that if we
want to get `dx` and `dy` for some `f(x,y)`, we have to rerun the entire computation
for both inputs. Compare this with reverse-mode autodiff where we can calculate arbitrary
derivatives for a function.

## How to use?
Usage is simple. Just replace every occurence of your standard data type `T`(like `double`)
with a `wfwdiff:var<T, T>`. A simple example:

```
// Original function
double f(double x) {
	return std::sin(x)+x*1.4;
}

// Autodiff'able function
using scalar_t = wfwdiff::var<double, double>;
scalar_t f(scalar_t x) {
	return std::sin(x)+x*1.4;
}
```

To get a derivative of `f` with regard to `x`, we can now simply run

```
int main() {
	scalar_t x = 2.3;

	const auto ans =
		wfwdiff::eval(f, wfwdiff::wrt(x), wfwdiff::at(x));

	return 0;
}
```

This will calculate the derivative of `f` at `x=2.3` with regard to `x`.

## Vectorized mode
WFWDiff also implements another mode that is similar to the above, but adds some
niceties to make usage for multiple variables more ergonomic. Instead of using a
`scalar_t` as defined above, we now use a vector type. This looks like follows:

```
using vec_t = wfwdiff::var<val_t, wfwdiff::vector<val_t, 2>>;
vec_t f_vec(vec_t x, vec_t y) {
    return std::exp(x) * std::cos(y);
}

int main() {
	scalar_t x = 3.2;
	scalar_t y = 2.1;

    const auto ans =
        wfwdiff::eval(f_vec, wfwdiff::parallelWrt(x, y), wfwdiff::at(x, y));

	auto dx = ans.grad[0];
	auto dy = ans.grad[1];

	return 0;
}
```

This is pretty much identical to the above with the difference of the `wfwdiff::parallelWrt`
call. This instructs WFWDiff to calculate derivatives for multiple variables in parallel.

Currently, this is only implemented as a sequential process behind the scenes. But it would be
trivial to use SIMD to parallelize the routines and get potentially `8 double` derivatives
per computation.
