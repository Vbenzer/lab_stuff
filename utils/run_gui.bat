REM @echo off
REM Change to script’s directory
cd /d %~dp0\..

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Run the Python script
python gui\main_window.py

REM Deactivate the virtual environment (optional)
deactivate