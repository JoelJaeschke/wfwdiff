# Testing library
FetchContent_Declare(
        catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.7)
FetchContent_MakeAvailable(catch)

# Tests need to be added as executables first
add_executable(test_scalar_diff test_scalar_autodiff.cpp)
target_compile_features(test_scalar_diff PRIVATE cxx_std_20)
target_compile_options(test_scalar_diff PRIVATE -mavx2 -march=native)
target_link_libraries(test_scalar_diff PRIVATE libwfwdiff Catch2::Catch2)

add_executable(test_vectorised_diff test_vectorised_diff.cpp)
target_compile_features(test_vectorised_diff PRIVATE cxx_std_20)
target_compile_options(test_vectorised_diff PRIVATE -mavx2 -march=native)
target_link_libraries(test_vectorised_diff PRIVATE libwfwdiff Catch2::Catch2)

add_test(NAME TestScalarDiff COMMAND test_scalar_diff)
add_test(NAME TestVectorDiff COMMAND test_vectorised_diff)
