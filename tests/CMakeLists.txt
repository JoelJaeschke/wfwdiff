# Testing library
FetchContent_Declare(
        catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.7)
FetchContent_MakeAvailable(catch)

# Tests need to be added as executables first
add_executable(test_autodiff test_autodiff.cpp)
target_compile_features(test_autodiff PRIVATE cxx_std_20)
target_compile_options(test_autodiff PRIVATE -mavx2 -march=native)
target_link_libraries(test_autodiff PRIVATE libwfwdiff Catch2::Catch2)

add_test(NAME TestAutodiff COMMAND test_autodiff)