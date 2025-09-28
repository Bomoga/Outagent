@echo off
setlocal

REM Adjust these if your paths differ
set PY=%~dp0venv\Scripts\python.exe
set UVICORN=backend.app.main:app
set PORT=8000

REM Optional: environment overrides (edit as needed)
set OUTAGENT_API_BASE=http://127.0.0.1:%PORT%
REM set EIA_API_KEY=PUT_YOUR_DEMO_KEY_HERE
REM set WX_LAT=26.5225
REM set WX_LON=-81.1637

echo Starting Outagent server on port %PORT% ...
"%PY%" -m uvicorn %UVICORN% --host 127.0.0.1 --port %PORT% --log-level warning