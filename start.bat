@echo off
cd /d "%~dp0"
title ACE-Step Workbench
if not exist ".venv\Scripts\python.exe" (
  echo  Not installed yet. Running install.bat first...
  call install.bat || exit /b 1
)
.venv\Scripts\python.exe -m workbench --open %*
if errorlevel 1 pause
