add_executable(gradient_descent autodiff_gradient_descent.cpp)

target_compile_features(gradient_descent PRIVATE cxx_std_20)
target_link_libraries(gradient_descent PRIVATE libwfwdiff xsimd)