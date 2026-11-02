@echo off
setlocal EnableDelayedExpansion

echo ================================================================================
echo SPHARE-BD: Reproducible Research Pipeline Launcher
echo IEEE Signal Processing Letters (2026)
echo Authors: Md. Sulyman Islam Sifat, Md Alamgir Kabir
echo ================================================================================
echo(

cd /d "%~dp0"

REM 1. Try python command in PATH
set "PY_CMD="
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
    goto :FOUND_PYTHON
)

REM 2. Try Python Launcher py -3
py -3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py -3"
    goto :FOUND_PYTHON
)

REM 3. Try standard Windows user install path
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :FOUND_PYTHON
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :FOUND_PYTHON
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :FOUND_PYTHON
)

echo [ERROR] Python 3.9+ was not found on your system PATH or standard directories.
echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
echo Make sure to check "Add Python to PATH" during installation.
echo(
pause
exit /b 1

:FOUND_PYTHON
echo [System] Using Python interpreter: !PY_CMD!
echo [System] Starting SPHARE-BD research reproduction pipeline...
echo(

!PY_CMD! run_all.py %*

set "EXIT_CODE=%ERRORLEVEL%"
if %EXIT_CODE% neq 0 (
    echo(
    echo [ERROR] SPHARE-BD reproduction pipeline encountered an error ^(Code %EXIT_CODE%^).
) else (
    echo(
    echo ================================================================================
    echo [SUCCESS] SPHARE-BD reproduction pipeline completed successfully!
    echo All results, figures, and reports are available in the 'results/' directory.
    echo ================================================================================
)

echo(
pause
exit /b %EXIT_CODE%
