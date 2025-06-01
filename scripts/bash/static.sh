#!/usr/bin/env bash
# ================================================================================
# - File:    static.sh
# - Purpose: Configure and build the csalt++ project as a static library
# ================================================================================

cmake -S ../../csalt++/ -B ../../csalt++/build/static/ -DBUILD_STATIC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../../csalt++/build/static/
# ================================================================================
# ================================================================================ 
# eof

