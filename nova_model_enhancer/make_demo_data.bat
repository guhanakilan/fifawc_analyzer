@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment is missing. Run setup.bat first.
  pause
  exit /b 1
)

echo Generating synthetic demo data into demo_data\ ...
echo This trains a small model, so it takes a minute.
echo.
.venv\Scripts\python.exe tools\make_demo_data.py || goto :failed

echo.
echo Done. Start the application with start.bat, then feed it these three files.
pause
exit /b 0

:failed
echo.
echo Demo data generation failed. Scroll up for the error.
pause
exit /b 1
