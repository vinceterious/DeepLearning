
################################
# Unit Tests
################################
# Add test cpp file
add_executable( layerUnitTest ut/TestLayer.cpp )
# Link test executable against gtest & gtest_main
target_link_libraries(layerUnitTest gtest gtest_main)