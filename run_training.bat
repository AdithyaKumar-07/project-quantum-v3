@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%\src;%PYTHONPATH%
.venv\Scripts\python.exe scripts\train_msd.py --max-patients 0 --resolution all
