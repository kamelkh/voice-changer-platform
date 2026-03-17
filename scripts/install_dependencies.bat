@echo off
echo ============================================
echo  Voice Changer Platform - Dependency Setup
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies.
    pause
    exit /b 1
)

echo.
echo [2/4] Creating required directories...
if not exist "models" mkdir models
if not exist "config\profiles" mkdir config\profiles
if not exist "logs" mkdir logs

echo.
echo [3/4] Checking CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo [4/4] Installation complete!
echo.
echo ============================================
echo  IMPORTANT: External Dependencies Required
echo ============================================
echo.
echo Please install VB-Audio Virtual Cable:
echo   https://vb-audio.com/Cable/
echo.
echo This is required to route processed audio to
echo apps like WhatsApp Web, Zoom, and Discord.
echo.
echo After installation:
echo   1. Run scripts\run.bat to start the app
echo   2. Select your microphone as input device
echo   3. Select "CABLE Input (VB-Audio)" as output
echo   4. In WhatsApp/Zoom/Discord, select
echo      "CABLE Output (VB-Audio)" as microphone
echo.
pause
