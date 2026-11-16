@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\train_unet_msd.py %*
) else (
  python scripts\train_unet_msd.py %*
)
if errorlevel 1 exit /b %errorlevel%
