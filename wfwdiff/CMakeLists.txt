file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/wfwdiff/**/*.hpp")

add_library(libwfwdiff INTERFACE ${HEADER_LIST})

target_include_directories(libwfwdiff INTERFACE ..)
target_compile_options(libwfwdiff INTERFACE -Wall -Werror -Wpedantic -O2)
target_compile_features(libwfwdiff INTERFACE cxx_std_20)
