@echo off
REM ================================================================================
REM - File:    static.bat
REM - Purpose: Configure and build the csalt++ project as a static library on Windows
REM ================================================================================

cmake -S ..\..\csalt++\ -B ..\..\csalt++\build\static\ -DBUILD_STATIC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ..\..\csalt++\build\static\

