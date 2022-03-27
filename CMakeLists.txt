cmake_minimum_required(VERSION 3.18)
project(WFwDiff
        VERSION 0.0.1
        DESCRIPTION "Vectorised automatic differentiation for C++"
        LANGUAGES CXX)

include(FetchContent)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(wfwdiff)
add_subdirectory(examples)