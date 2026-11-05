@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Install Python 3.12 with the Python launcher, then run this file again.
  goto done
)
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check the error above and your internet connection.
  goto done
)
echo Setup complete. Run START_WINDOWS.bat to open Gesture Lab.
:done
pause
