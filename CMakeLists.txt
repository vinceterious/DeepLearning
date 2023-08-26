cmake_minimum_required(VERSION 3.14)
project(deeplearning)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################
# GTest
################################
ADD_SUBDIRECTORY (googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Dataset example
################################

file(COPY dataset/winequality-white.csv DESTINATION "${CMAKE_BINARY_DIR}")

################################
# Exec
################################
add_subdirectory(matrix)
add_executable(main main.cpp )
target_link_libraries(main matrix)