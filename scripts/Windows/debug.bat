@echo off
REM ================================================================================
REM - File:    debug.bat
REM - Purpose: Configure and build the csalt++ project in Debug mode on Windows
REM ================================================================================

cmake -S ..\..\csalt++\ -B ..\..\csalt++\build\debug\ -DCMAKE_BUILD_TYPE=Debug
cmake --build ..\..\csalt++\build\debug\

