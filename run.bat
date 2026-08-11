@echo off
setlocal EnableDelayedExpansion

title Liver Cirrhosis Detection System

echo.
echo  ============================================================
echo   Liver Cirrhosis Detection System — Dev Server Launcher
echo  ============================================================
echo.

set ROOT=%~dp0
set BACKEND=%ROOT%backend
set FRONTEND=%ROOT%frontend
set VENV=%ROOT%venv

:: ── 1. Check Python ───────────────────────────────────────────────────────────
where python3.14 >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON=python3.14
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 (
        set PYTHON=python
    ) else (
        echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
        pause & exit /b 1
    )
)
echo [OK] Python : %PYTHON%

:: ── 2. Check Node / npm ───────────────────────────────────────────────────────
where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm not found. Install Node.js 18+ from https://nodejs.org
    pause & exit /b 1
)
echo [OK] npm    : found

:: ── 3. Create + activate virtual environment ──────────────────────────────────
if not exist "%VENV%\Scripts\activate.bat" (
    echo.
    echo [SETUP] Creating Python virtual environment...
    %PYTHON% -m venv "%VENV%"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo [OK] Virtual environment created.
)

:: ── 4. Install / update Python dependencies ───────────────────────────────────
echo.
echo [SETUP] Installing Python dependencies...
call "%VENV%\Scripts\activate.bat"
pip install --quiet --upgrade pip
pip install --quiet -r "%ROOT%requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip install failed. See output above.
    pause & exit /b 1
)
echo [OK] Python dependencies ready.

:: ── 5. Create .env if missing ─────────────────────────────────────────────────
if not exist "%ROOT%.env" (
    echo.
    echo [SETUP] .env not found — copying from .env.example...
    copy "%ROOT%.env.example" "%ROOT%.env" >nul
    echo [OK] .env created. Edit it before running in production.
)

:: ── 6. Install frontend npm packages ──────────────────────────────────────────
if not exist "%FRONTEND%\node_modules" (
    echo.
    echo [SETUP] Installing frontend npm packages...
    pushd "%FRONTEND%"
    npm install --silent
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] npm install failed.
        popd & pause & exit /b 1
    )
    popd
    echo [OK] Frontend packages ready.
)

:: ── 7. Launch backend in a new window ─────────────────────────────────────────
echo.
echo [START] Launching FastAPI backend on http://localhost:8000 ...
start "Backend — FastAPI" cmd /k "cd /d "%BACKEND%" && call "%VENV%\Scripts\activate.bat" && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: ── 8. Launch frontend in a new window ───────────────────────────────────────
echo [START] Launching React frontend on http://localhost:3000 ...
start "Frontend — React" cmd /k "cd /d "%FRONTEND%" && npm start"

echo.
echo  ============================================================
echo   Both servers are starting. Open your browser at:
echo     Frontend  →  http://localhost:3000
echo     API Docs  →  http://localhost:8000/docs
echo  ============================================================
echo.
echo  Press any key to exit this launcher (servers keep running).
pause >nul
