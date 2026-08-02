@echo off
rem Pulls the latest PRISM code and refreshes Python dependencies. Run from
rem anywhere -- this always operates on the checkout it lives in. Does not
rem touch the Desktop PRISM.bat shortcut (same repo location, so it's
rem still valid); re-run scripts\install.bat instead if the folder moved.
setlocal
cd /d "%~dp0.."

where git >nul 2>nul
if errorlevel 1 (
  echo git not found on PATH. Install Git for Windows, then re-run this script.
  pause
  exit /b 1
)

echo Pulling the latest PRISM code...
git pull
if errorlevel 1 (
  echo.
  echo git pull failed -- resolve any conflicts/errors above, then re-run.
  pause
  exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" not found on PATH. Install Python 3.11 from
  echo python.org, then re-run this script.
  pause
  exit /b 1
)

echo.
echo Refreshing Python dependencies ^(py -3.11^)...
py -3.11 -m pip install --upgrade -r requirements.txt
py -3.11 -m pip install --upgrade -r requirements-qt.txt
py -3.11 -m pip install --upgrade -r requirements-xas.txt
py -3.11 -m pip install --upgrade -r requirements-glass.txt
py -3.11 -m pip install --upgrade -r requirements-saxs-edf.txt
if errorlevel 1 (
  echo.
  echo FAILED -- see the pip output above for details.
  pause
  exit /b 1
)

echo.
echo Done. PRISM is up to date.
pause
