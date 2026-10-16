@echo off
cd /d "%~dp0"
rem Set WB_NO_PAUSE=1 to never wait for a key (for scripts and AI agents).
set "WB_PAUSE=pause"
if defined WB_NO_PAUSE set "WB_PAUSE=type nul"
title ACE-Step Workbench
if not exist ".venv\Scripts\python.exe" (
  echo  Not installed yet. Running install.bat first...
  call install.bat || exit /b 1
)
.venv\Scripts\python.exe -m workbench --open %*
if errorlevel 1 %WB_PAUSE%
