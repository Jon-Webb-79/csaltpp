# ================================================================================
# ================================================================================
# - File:    CMakeLists.txt
# - Purpose: Primary CMake file for C build system
#
# Source Metadata
# - Author:  Name
# - Date:    Month Day, Year
# - Version: 1.0
# - Copyright: Copyright Year, Name Inc.
# ================================================================================
# ================================================================================

cmake_minimum_required(VERSION 4.0.2)
project(csaltpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Option for static linking
option(BUILD_STATIC "Build static test executables" OFF)

# Platform-specific definitions
if(UNIX AND NOT APPLE)
    add_definitions(-D_GNU_SOURCE)
elseif(APPLE)
    add_definitions(-D_DARWIN_C_SOURCE)
elseif(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601 -DWIN32_LEAN_AND_MEAN)
endif()

# Compiler flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Configuring Debug build with SIMD disabled (except SSE3)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wpedantic -O0 -g -march=x86-64 -mno-avx -mno-avx2 -mno-fma -msse3 -msse4.1")
    else()
        message(STATUS "Configuring Release/custom build with SIMD enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wpedantic -O3 -march=native")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
    add_definitions(/wd4996 /D_CRT_SECURE_NO_WARNINGS)
endif()

# Include the header-only matrix library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Delegate test building to test/CMakeLists.txt
enable_testing()
add_subdirectory(test)
# ================================================================================
# ================================================================================
# eof
