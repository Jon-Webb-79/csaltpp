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

# --------------------------------------------------------------------
# Project
# --------------------------------------------------------------------
project(csaltpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# --------------------------------------------------------------------
# Options (scripts toggle these)
# --------------------------------------------------------------------
option(CSALTXX_BUILD_TESTS   "Build unit tests" OFF)
option(CSALTXX_BUILD_STATIC  "Also build a static library (only meaningful when sources exist)" OFF)

# Primary shared/static switch (shared default unless scripts override)
if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()

include(GNUInstallDirs)

# Uniform output layout for *all* targets (incl. tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --------------------------------------------------------------------
# Sources / Headers
# --------------------------------------------------------------------
# Add .cpp files here when you have them (e.g., src/matrix.cpp)
set(CSALTXX_SOURCES
    error.cpp
    allocator.cpp
)

set(CSALTXX_PUBLIC_INCLUDE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CSALTXX_PRIVATE_SIMD_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/simd)

# --------------------------------------------------------------------
# Library target (dual-mode)
#   - If CSALTXX_SOURCES is empty => INTERFACE (header-only)
#   - Else => normal library (STATIC/SHARED governed by BUILD_SHARED_LIBS)
# --------------------------------------------------------------------
list(LENGTH CSALTXX_SOURCES _csaltxx_src_count)

if(_csaltxx_src_count EQUAL 0)
  add_library(csaltpp INTERFACE)
else()
  add_library(csaltpp ${CSALTXX_SOURCES})
endif()

add_library(csaltpp::csaltpp ALIAS csaltpp)

# Include dirs:
# - include/ is public
# - simd/ is exposed because headers may include .inl from there
if(_csaltxx_src_count EQUAL 0)
  target_include_directories(csaltpp
    INTERFACE
      $<BUILD_INTERFACE:${CSALTXX_PUBLIC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/csaltpp>
      $<BUILD_INTERFACE:${CSALTXX_PRIVATE_SIMD_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/csaltpp/simd>
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
else()
  target_include_directories(csaltpp
    PUBLIC
      $<BUILD_INTERFACE:${CSALTXX_PUBLIC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/csaltpp>
      $<BUILD_INTERFACE:${CSALTXX_PRIVATE_SIMD_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/csaltpp/simd>
  )
endif()

# --------------------------------------------------------------------
# Warnings / SIMD flags
#   Header-only => INTERFACE flags (apply to consumers)
#   Compiled lib => PRIVATE flags (apply to library build)
# --------------------------------------------------------------------
if(_csaltxx_src_count EQUAL 0)
  set(_csaltxx_flag_scope INTERFACE)
else()
  set(_csaltxx_flag_scope PRIVATE)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(csaltpp ${_csaltxx_flag_scope} -Wall -Wextra -Wpedantic)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "csaltpp: Debug build with AVX/AVX2 disabled; SSE3/SSE4.1 allowed, no -Werror")
    target_compile_options(csaltpp ${_csaltxx_flag_scope}
      -O0 -g -march=x86-64 -mno-avx -mno-avx2 -mno-fma -msse3 -msse4.1
    )
  else()
    message(STATUS "csaltpp: ${CMAKE_BUILD_TYPE} build with -O3 -march=native and -Werror")
    target_compile_options(csaltpp ${_csaltxx_flag_scope} -O3 -march=native -Werror)
  endif()

elseif (MSVC)
  target_compile_options(csaltpp ${_csaltxx_flag_scope} /W4)
  target_compile_definitions(csaltpp ${_csaltxx_flag_scope} _CRT_SECURE_NO_WARNINGS)

  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(csaltpp ${_csaltxx_flag_scope} /arch:AVX2 /WX)
  endif()
endif()

# Version info only makes sense for a compiled library
if(NOT _csaltxx_src_count EQUAL 0)
  set_target_properties(csaltpp PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
endif()

# --------------------------------------------------------------------
# Optional additional static library (only when sources exist)
# --------------------------------------------------------------------
if(CSALTXX_BUILD_STATIC AND NOT _csaltxx_src_count EQUAL 0)
  add_library(csaltpp_static STATIC ${CSALTXX_SOURCES})
  add_library(csaltpp::csaltpp_static ALIAS csaltpp_static)
  set_target_properties(csaltpp_static PROPERTIES OUTPUT_NAME csaltpp)

  target_include_directories(csaltpp_static
    PUBLIC
      $<BUILD_INTERFACE:${CSALTXX_PUBLIC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/csaltpp>
      $<BUILD_INTERFACE:${CSALTXX_PRIVATE_SIMD_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/csaltpp/simd>
  )

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(csaltpp_static PRIVATE -Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(csaltpp_static PRIVATE
        -O0 -g -march=x86-64 -mno-avx -mno-avx2 -mno-fma -msse3 -msse4.1
      )
    else()
      target_compile_options(csaltpp_static PRIVATE -O3 -march=native -Werror)
    endif()
  elseif (MSVC)
    target_compile_options(csaltpp_static PRIVATE /W4)
    target_compile_definitions(csaltpp_static PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(csaltpp_static PRIVATE /arch:AVX2 /WX)
    endif()
  endif()
endif()

# --------------------------------------------------------------------
# Installation + CMake package config for find_package(csaltpp)
# --------------------------------------------------------------------
install(TARGETS csaltpp
  EXPORT csaltppTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(CSALTXX_BUILD_STATIC AND NOT _csaltxx_src_count EQUAL 0)
  install(TARGETS csaltpp_static
    EXPORT csaltppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

# Public headers → include/csaltpp/
install(DIRECTORY ${CSALTXX_PUBLIC_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/csaltpp
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# SIMD inlines → include/csaltpp/simd/
install(DIRECTORY ${CSALTXX_PRIVATE_SIMD_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/csaltpp/simd
  FILES_MATCHING PATTERN "*.inl"
)

install(EXPORT csaltppTargets
  FILE csaltppTargets.cmake
  NAMESPACE csaltpp::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/csaltpp
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/csaltppConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/csaltppConfig.cmake"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/csaltppTargets.cmake\")\n")

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/csaltppConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/csaltppConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/csaltpp
)

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
if(CSALTXX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# ================================================================================
# ================================================================================
# eof
