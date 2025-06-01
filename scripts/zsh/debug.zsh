#!/usr/bin/zsh
# ================================================================================
# - File:    debug.zsh
# - Purpose: Build C++ version of the csalt++ project in Debug mode using CMake
#
# Source Metadata
# - Author:  Jonathan A. Webb
# - Date:    June 1, 2025
# - Version: 1.0
# - Copyright: Copyright 2025, Jon Webb Inc.
# ================================================================================

cmake -S ../../csalt++/ -B ../../csalt++/build/debug/ -DCMAKE_BUILD_TYPE=Debug
cmake --build ../../csalt++/build/debug/
# ================================================================================ 
# ================================================================================ 
# eof
