@echo off
rem One-command portable-exe release: PyInstaller build with the exclusion
rem list this app needs (the shared Python environment contains torch,
rem transformers, llvmlite, PyQt5 and other heavyweights that would
rem otherwise be silently bundled - the first unfiltered build was 883 MB
rem vs 349 MB with these exclusions), then zip dist\PRISM into a single
rem shareable archive.
rem
rem Note: the exe excludes xraylarch AND glasspy; Larch-dependent XAS
rem steps (normalization/EXAFS) and glasspy-dependent GlassNet
rem predictions both need a Python install + PRISM.bat instead. glasspy
rem is excluded (not just its torch dependency) because a build that
rem tried excluding torch/pyarrow/dask directly still silently bundled
rem torch (246 MB) via a transitive glasspy->lightning PyInstaller hook
rem that collects torch's compiled .dll files directly, bypassing
rem --exclude-module torch entirely - excluding glasspy itself removes
rem the whole causal chain instead of chasing individual transitive deps.
cd /d "%~dp0"

py -3.11 -m PyInstaller --noconfirm --clean --windowed --name PRISM ^
  --icon assets\prism.ico --add-data "assets;assets" ^
  --exclude-module larch --exclude-module glasspy ^
  --exclude-module wx --exclude-module tkinter ^
  --exclude-module PyQt5 --exclude-module PyQt6 ^
  --exclude-module IPython --exclude-module jupyter --exclude-module nbformat ^
  --exclude-module notebook --exclude-module zmq ^
  --exclude-module torch --exclude-module transformers --exclude-module tokenizers ^
  --exclude-module llvmlite --exclude-module numba ^
  --exclude-module botocore --exclude-module boto3 ^
  --exclude-module h5py --exclude-module lxml ^
  --exclude-module cryptography --exclude-module paramiko ^
  --exclude-module pyarrow --exclude-module dask ^
  --exclude-module lightning --exclude-module pytorch_lightning ^
  qt_main.py
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo Copying no-Python database download scripts and LICENSE ...
copy /y scripts\Download-RRUFF-database.bat dist\PRISM\ >nul
copy /y scripts\Download-RRUFF-database.ps1 dist\PRISM\ >nul
copy /y scripts\Download-AMCSD-structures.bat dist\PRISM\ >nul
copy /y LICENSE dist\PRISM\LICENSE.txt >nul

echo Zipping dist\PRISM ...
powershell -NoProfile -Command "Compress-Archive -Path 'dist\PRISM' -DestinationPath 'dist\PRISM-portable.zip' -Force"
echo Done: dist\PRISM-portable.zip
pause
