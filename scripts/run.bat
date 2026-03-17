@echo off
echo Starting Voice Changer Platform...
cd /d "%~dp0.."
python -m src.main %*
if errorlevel 1 (
    echo.
    echo ERROR: Application exited with an error.
    echo Check logs\ directory for details.
    pause
)
