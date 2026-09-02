@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  NoVA Model Enhancer - setup
echo ============================================
echo.

echo [1/4] Checking Python...
python --version || goto :python_error
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || goto :python_version_error

echo [2/4] Creating the virtual environment...
if not exist ".venv\Scripts\python.exe" python -m venv .venv || goto :venv_error

echo [3/4] Installing backend packages...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :install_error

echo [4/4] Installing frontend packages...
if not exist "frontend\package.json" goto :frontend_error
pushd frontend
call npm.cmd install || (popd & goto :npm_error)
popd

echo.
echo Setup complete. Double-click start.bat to launch the application.
echo Nothing outside this folder was modified.
pause
exit /b 0

:python_error
echo.
echo Python was not found on PATH.
echo Install Python 3.10 or newer with "Add python.exe to PATH" ticked, then rerun setup.bat.
pause
exit /b 1

:python_version_error
echo.
echo This Python is too old. Version 3.10 or newer is required; yours is:
python --version
echo.
echo Install a newer Python from https://www.python.org/downloads/ with
echo "Add python.exe to PATH" ticked, then rerun setup.bat.
pause
exit /b 1

:venv_error
echo.
echo The virtual environment could not be created. Check that this folder is writable.
pause
exit /b 1

:install_error
echo.
echo Backend package installation failed. Scroll up for the pip error.
echo A proxy or an offline machine is the usual cause.
pause
exit /b 1

:frontend_error
echo.
echo frontend\package.json is missing - this copy of the application is incomplete.
pause
exit /b 1

:npm_error
echo.
echo Frontend package installation failed. Check that Node.js 18 or newer is on PATH.
pause
exit /b 1
