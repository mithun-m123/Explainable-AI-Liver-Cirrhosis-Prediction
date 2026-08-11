#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Liver Cirrhosis Detection System — Dev Server Launcher (Linux/macOS)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$ROOT/venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[SETUP]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo ""
echo "  ============================================================"
echo "   Liver Cirrhosis Detection System — Dev Server Launcher"
echo "  ============================================================"
echo ""

# ── 1. Check Python ───────────────────────────────────────────────────────────
if   command -v python3.14 &>/dev/null; then PYTHON=python3.14
elif command -v python3    &>/dev/null; then PYTHON=python3
elif command -v python     &>/dev/null; then PYTHON=python
else err "Python not found. Install Python 3.10+ from https://python.org"; fi
info "Python  : $($PYTHON --version)"

# ── 2. Check Node / npm ───────────────────────────────────────────────────────
command -v npm &>/dev/null || err "npm not found. Install Node.js 18+ from https://nodejs.org"
info "npm     : $(npm --version)"

# ── 3. Virtual environment ────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
    warn "Creating Python virtual environment..."
    $PYTHON -m venv "$VENV"
    info "Virtual environment created."
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 4. Install Python dependencies ────────────────────────────────────────────
warn "Installing / verifying Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT/requirements.txt"
info "Python dependencies ready."

# ── 5. Create .env if missing ─────────────────────────────────────────────────
if [[ ! -f "$ROOT/.env" ]]; then
    warn ".env not found — copying from .env.example..."
    cp "$ROOT/.env.example" "$ROOT/.env"
    info ".env created. Edit it before deploying to production."
fi

# ── 6. Install frontend npm packages ──────────────────────────────────────────
if [[ ! -d "$FRONTEND/node_modules" ]]; then
    warn "Installing frontend npm packages..."
    (cd "$FRONTEND" && npm install --silent)
    info "Frontend packages ready."
fi

# ── 7. Launch backend ─────────────────────────────────────────────────────────
echo ""
info "Starting FastAPI backend  →  http://localhost:8000"
(cd "$BACKEND" && uvicorn main:app --host 0.0.0.0 --port 8000 --reload) &
BACKEND_PID=$!

# Small delay so the backend port is ready before the browser opens
sleep 2

# ── 8. Launch frontend ────────────────────────────────────────────────────────
info "Starting React frontend   →  http://localhost:3000"
(cd "$FRONTEND" && npm start) &
FRONTEND_PID=$!

echo ""
echo "  ============================================================"
echo "   Servers are running:"
echo "     Frontend  →  http://localhost:3000"
echo "     API Docs  →  http://localhost:8000/docs"
echo "   Press Ctrl+C to stop both servers."
echo "  ============================================================"
echo ""

# ── 9. Trap Ctrl+C and kill both child processes ──────────────────────────────
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
