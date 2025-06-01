@echo off
REM ================================================================================
REM - File:    install.bat
REM - Purpose: Install csalt++ headers into system-level include folder (for dev use)
REM ================================================================================

REM Check for administrator rights (requires powershell)
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Please run this script as Administrator.
    exit /b 1
)

REM Define install paths
set INCLUDE_DIR=%ProgramFiles%\csaltpp\include
set BACKUP_DIR=%TEMP%\csaltpp_backup

REM Create directories
mkdir "%INCLUDE_DIR%" >nul 2>&1
mkdir "%BACKUP_DIR%" >nul 2>&1

REM Install each .hpp file
for %%F in (..\..\csalt++\include\*.hpp) do (
    set "SRC=%%F"
    set "DEST=%INCLUDE_DIR%\%%~nxF"
    setlocal EnableDelayedExpansion

    if exist "!DEST!" (
        echo Updating existing %%~nxF...
        set "BACKUP_FILE=%BACKUP_DIR%\%%~nxF_%DATE:/=-%_%TIME::=-%"
        copy /Y "!DEST!" "!BACKUP_FILE!" >nul
        echo Backed up to !BACKUP_FILE!
    ) else (
        echo Installing new %%~nxF...
    )

    copy /Y "%%F" "!DEST!" >nul
    if %errorlevel% EQU 0 (
        echo %%~nxF installed successfully.
    ) else (
        echo Failed to install %%~nxF
        exit /b 1
    )
    endlocal
)

echo.
echo Installation/Update completed successfully.
echo Backups (if any) are stored in %BACKUP_DIR%

