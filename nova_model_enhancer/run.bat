@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
if not exist "frontend\node_modules" (
  echo The frontend packages are missing. Run setup.bat first.
  pause
  exit /b 1
)

REM Both windows stay open so their logs remain readable. Close them to stop.
start "NoVA Enhancer API" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8081"
start "NoVA Enhancer UI"  cmd /k "cd /d "%~dp0frontend" && npm.cmd run dev"

echo Waiting for the backend to come up...
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:5174

echo.
echo   UI       http://127.0.0.1:5174
echo   API      http://127.0.0.1:8081
echo   API docs http://127.0.0.1:8081/docs
echo.
echo Close the two console windows to stop the application.
endlocal
