@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Run setup_windows.bat first.
  exit /b 1
)
.venv\Scripts\python.exe scripts\validate_release.py
exit /b %errorlevel%
