#ifndef WFWDIFF_UTIL_HPP
#define WFWDIFF_UTIL_HPP

#include <cstdint>

namespace wfwdiff {
namespace generic_vec {

template<std::size_t requested>
struct fastest_vec_available;

template<>
struct fastest_vec_available<1> {
    static const std::size_t size = 4;
};

template<>
struct fastest_vec_available<2> {
    static const std::size_t size = 4;
};

template<>
struct fastest_vec_available<3> {
    static const std::size_t size = 4;
};

template<>
struct fastest_vec_available<4> {
    static const std::size_t size = 4;
};

} // namespace generic_vec
} // namespace wfwdiff


#endif  // WFWDIFF_UTIL_HPP
