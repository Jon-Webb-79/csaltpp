@echo off
REM ================================================================================
REM Debug build driver for csalt++ (Windows version)
REM - Configures Debug build with CSALTXX_BUILD_TESTS=ON
REM - Produces: build\debug\bin\unit_tests.exe
REM - Runs unit_tests.exe (use "debug.bat --no-run" to skip)
REM
REM Usage:
REM   debug.bat
REM   debug.bat --no-run
REM   debug.bat --clean
REM   debug.bat --generator "Visual Studio 17 2022"
REM ================================================================================

setlocal enabledelayedexpansion

REM --------------------- Defaults ---------------------
set BUILD_DIR=build\debug
set BUILD_TYPE=Debug
set RUN_AFTER_BUILD=1
set GEN=
set CLEAN=0

REM --------------------- Parse args -------------------
:parse_args
if "%~1"=="" goto args_done

if "%~1"=="--no-run" (
    set RUN_AFTER_BUILD=0
    shift
    goto parse_args
)

if "%~1"=="--clean" (
    set CLEAN=1
    shift
    goto parse_args
)

if "%~1"=="--generator" (
    set GEN=%~2
    shift
    shift
    goto parse_args
)

if "%~1"=="-h" (
    echo Usage: debug.bat [--no-run] [--clean] [--generator "NAME"]
    exit /b 0
)

if "%~1"=="--help" (
    echo Usage: debug.bat [--no-run] [--clean] [--generator "NAME"]
    exit /b 0
)

echo Unknown arg: %~1
exit /b 1

:args_done

REM --------------------- Paths ------------------------
set SCRIPT_DIR=%~dp0
REM Go up two levels from scripts\Windows\ to project root
pushd "%SCRIPT_DIR%\..\.."
set PROJ_ROOT=%CD%
popd

set SRC_DIR=%PROJ_ROOT%\csalt++
set OUT_DIR=%PROJ_ROOT%\%BUILD_DIR%
set BIN_DIR=%OUT_DIR%\bin
set UNIT_EXE=%BIN_DIR%\unit_tests.exe

REM --------------------- Clean if requested -----------
if %CLEAN%==1 (
    echo Cleaning %OUT_DIR%
    rmdir /s /q "%OUT_DIR%" 2>nul
)

REM --------------------- Configure --------------------
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set CMAKE_ARGS=-S "%SRC_DIR%" -B "%OUT_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_SHARED_LIBS=ON -DCSALTXX_BUILD_TESTS=ON

if not "%GEN%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -G "%GEN%"
)

echo ==^> Configuring with: cmake %CMAKE_ARGS%
cmake %CMAKE_ARGS%
if errorlevel 1 exit /b 1

REM --------------------- Build ------------------------
echo ==^> Building unit_tests
cmake --build "%OUT_DIR%" --config %BUILD_TYPE% --target unit_tests
if errorlevel 1 exit /b 1

REM --------------------- Locate exe -------------------
if exist "%UNIT_EXE%" (
    echo ==^> Built: %UNIT_EXE%
) else (
    echo WARNING: unit_tests.exe not found in expected bin\ directory.
    for /r "%OUT_DIR%" %%f in (unit_tests.exe) do (
        set UNIT_EXE=%%f
        echo ==^> Found at: %%f
        goto exe_found
    )
    echo ERROR: unit_tests.exe not found.
    exit /b 2
)
:exe_found

REM --------------------- Run it -----------------------
if %RUN_AFTER_BUILD%==1 (
    echo ==^> Running unit_tests
    "%UNIT_EXE%"
)

echo ==^> Debug build complete.
exit /b 0

