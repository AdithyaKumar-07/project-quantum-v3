@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Run setup_windows.bat first.
  exit /b 1
)
set PYTHONPATH=%CD%\src;%PYTHONPATH%
.venv\Scripts\python.exe -m uvicorn qmi.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
