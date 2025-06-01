include(FetchContent)

# Fetch GoogleTest if not already present
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

# Define test executable
add_executable(test_matrix test_matrix.cpp)

# Link with GoogleTest
target_link_libraries(test_matrix PRIVATE gtest_main)

# Make sure the headers are available
target_include_directories(test_matrix PRIVATE
    ${CMAKE_SOURCE_DIR}/csalt++/include
)

# Auto-register with CTest
include(GoogleTest)
gtest_discover_tests(test_matrix)

