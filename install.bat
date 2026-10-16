@echo off
setlocal
cd /d "%~dp0"
rem Set WB_NO_PAUSE=1 to never wait for a key (for scripts and AI agents).
set "WB_PAUSE=pause"
if defined WB_NO_PAUSE set "WB_PAUSE=type nul"
title ACE-Step Workbench - install
echo.
echo  === ACE-Step Workbench installer ===
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo  Installing uv ^(fast Python package manager^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo  Could not install uv. Get it from https://docs.astral.sh/uv/ and run this again.
  %WB_PAUSE% & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo  Creating the Workbench environment...
  uv venv .venv --python 3.12 || (%WB_PAUSE% & exit /b 1)
)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || (%WB_PAUSE% & exit /b 1)

.venv\Scripts\python.exe install.py %*
if errorlevel 1 (%WB_PAUSE% & exit /b 1)
echo.
echo  Done. Double-click start.bat to launch.
%WB_PAUSE%
