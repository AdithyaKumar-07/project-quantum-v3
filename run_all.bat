@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\ensure_api.py
if errorlevel 1 (
  echo API startup/verification failed. Streamlit was not started against an unknown backend.
  pause
  exit /b 1
)
set PYTHONPATH=%CD%\src;%PYTHONPATH%
.venv\Scripts\python.exe -m streamlit run app\Home.py
