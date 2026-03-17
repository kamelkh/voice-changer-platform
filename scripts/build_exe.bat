@echo off
:: ============================================================================
::  build_exe.bat  –  Build VoiceChanger.exe with PyInstaller
::
::  Prerequisites:
::    1. Python 3.10+ in PATH  (or run from the project venv)
::    2. All dependencies installed:  pip install -r requirements.txt
::    3. (Optional) UPX installed and in PATH for smaller output binary
::
::  Output:  dist\VoiceChanger.exe
:: ============================================================================

setlocal enabledelayedexpansion

:: Move to the project root (one level up from scripts\)
cd /d "%~dp0.."

echo ========================================
echo  AI Voice Changer Platform  –  EXE Build
echo ========================================
echo.

:: ── 1. Install / upgrade PyInstaller ─────────────────────────────────────────
echo [1/4] Installing PyInstaller...
python -m pip install --upgrade pyinstaller
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

:: ── 2. Clean previous build artefacts ────────────────────────────────────────
echo.
echo [2/4] Cleaning previous build...
if exist build     rmdir /s /q build
if exist dist      rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

:: ── 3. Run PyInstaller ────────────────────────────────────────────────────────
echo.
echo [3/4] Running PyInstaller...
python -m PyInstaller voice_changer.spec --clean --noconfirm --log-level WARN
if errorlevel 1 (
    echo ERROR: PyInstaller failed – see output above.
    pause
    exit /b 1
)

:: ── 4. Done ───────────────────────────────────────────────────────────────────
echo.
echo [4/4] Build complete!
echo.
echo   Executable:  dist\VoiceChanger.exe
echo.
echo   To run:
echo     dist\VoiceChanger.exe
echo.
echo ========================================

:: Open the dist folder in Explorer
start "" explorer "%~dp0..\dist"

pause
