include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests
  test_error.cpp
  test_allocator.cpp
  test_utilities.cpp
  test_pointers.cpp
  test_string.cpp
  test_array.cpp
  #  test_matrix.cpp
)

target_link_libraries(unit_tests
  PRIVATE
    csaltpp::csaltpp
    GTest::gtest
    GTest::gtest_main
)

# Optional: warnings for test TU
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
elseif (MSVC)
  target_compile_options(unit_tests PRIVATE /W4)
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)

