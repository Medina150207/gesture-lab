@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app.py
  goto done
)
python -c "import streamlit,torch,torchvision,pandas,PIL" >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.12 and run SETUP_WINDOWS.bat first.
  goto done
)
python -m streamlit run app.py
:done
pause
