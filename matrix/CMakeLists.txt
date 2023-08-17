add_library(matrix STATIC Matrix.cpp)

################################
# Unit Tests
################################
# Add test cpp file
add_executable( matrixUnitTest ut/TestMatrix.cpp )
# Link test executable against gtest & gtest_main
target_link_libraries(matrixUnitTest gtest gtest_main)