file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/wfwdiff/*.hpp")

add_library(libwfwdiff INTERFACE)

target_include_directories(libwfwdiff INTERFACE ..)
target_compile_options(libwfwdiff INTERFACE -Wall -Werror -Wpedantic -O2 -march=native)
target_compile_features(libwfwdiff INTERFACE cxx_std_20)

FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
        GIT_TAG 8.0.5)
FetchContent_MakeAvailable(xsimd)