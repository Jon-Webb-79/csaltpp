#!/usr/bin/zsh
# ================================================================================
# - File:    static.zsh
# - Purpose: This file configures and builds a static library version of csalt++
#
# Source Metadata
# - Author:  Jonathan A. Webb
# - Date:    June 1, 2025
# - Version: 1.0
# - Copyright: Copyright 2025, Jon Webb Inc.
# ================================================================================

cmake -S ../../csalt++/ -B ../../csalt++/build/static/ -DBUILD_STATIC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../../csalt++/build/static/
# ================================================================================
# ================================================================================ 
# eof

