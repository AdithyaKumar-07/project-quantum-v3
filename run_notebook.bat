@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Run setup_windows.bat first.
  exit /b 1
)
set PYTHONPATH=%CD%\src;%PYTHONPATH%
.venv\Scripts\python.exe -m jupyter lab notebooks
