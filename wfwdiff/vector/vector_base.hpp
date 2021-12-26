#ifndef WFWDIFF_VECTOR_BASE_H
#define WFWDIFF_VECTOR_BASE_H

#include <array>
#include <algorithm>
#include <functional>
#include <cassert>
#include <iostream>

namespace wfwdiff {
namespace generic_vec {

template <typename T, size_t width>
struct vector {
   private:
    std::array<T, width> storage_;

   public:
    vector(): storage_() {};
    vector(const std::array<T, width>& input): storage_(input) {};
    vector(const T initializer) {
        std::fill_n(storage_.begin(), width, initializer);
    };
    vector(const vector<T, width>& vec): storage_(vec.data()) {};

    ~vector() = default;

    auto operator+(const vector<T, width>& rhs) const {
        std::array<T, width> result{};
        std::transform(storage_.begin(), storage_.end(),
            rhs.data().begin(), result.begin(),
            std::plus<T>());

        return vector(result);
    };

    auto operator-(const vector<T, width>& rhs) const {
        std::array<T, width> result{};
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), result.begin(),
                       std::minus<T>());

        return vector(result);
    };

    auto operator*(const vector<T, width>& rhs) const {
        std::array<T, width> result{};
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), result.begin(),
                       std::multiplies<T>());

        return vector(result);
    };

    auto operator/(const vector<T, width>& rhs) const {
        std::array<T, width> result{};
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), result.begin(),
                       std::divides<T>());

        return vector(result);
    };

    vector<T, width>& operator+=(const vector<T, width>& rhs) {
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), storage_.begin(),
                       std::plus<T>());

        return *this;
    };

    vector<T, width>& operator-=(const vector<T, width>& rhs) {
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), storage_.begin(),
                       std::minus<T>());

        return *this;
    };

    vector<T, width>& operator*=(const vector<T, width>& rhs) {
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), storage_.begin(),
                       std::multiplies<T>());

        return *this;
    };

    vector<T, width>& operator/=(const vector<T, width>& rhs) {
        std::transform(storage_.begin(), storage_.end(),
                       rhs.data().begin(), storage_.begin(),
                       std::divides<T>());

        return *this;
    };

    const T& operator[](const size_t idx) const {
        return storage_[idx];
    }

    T& operator[](const size_t idx) {
        return storage_[idx];
    }

    const std::array<T, width>& data() const {
        return storage_;
    };

    std::array<T, width>& data() {
        return storage_;
    };
};

}  // namespace generic_vec
}  // namespace wfwdiff

namespace std {

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> sin(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::sin);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> cos(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::cos);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> tan(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::tan);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> asin(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::asin);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> acos(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::acos);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> atan(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::atan);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> exp(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::exp);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width> log(const wfwdiff::generic_vec::vector<T, width>& x) {
    std::array<T, width> result = x.data();
    std::for_each(result.begin(), result.end(), std::log);

    return wfwdiff::generic_vec::vector<T, width>(result);
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& sin(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::sin);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& cos(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::cos);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& tan(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::tan);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& asin(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::asin);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& acos(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::acos);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& atan(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::atan);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& exp(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::exp);

    return x;
}

template<typename T, size_t width>
wfwdiff::generic_vec::vector<T, width>& log(wfwdiff::generic_vec::vector<T, width>& x) {
    std::for_each(x.data().begin(), x.data().end(), std::log);

    return x;
}

} // namespace std

#endif  // WFWDIFF_VECTOR_BASE_H
