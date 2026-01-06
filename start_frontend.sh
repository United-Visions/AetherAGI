#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
cd frontend_flask
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           AetherMind Frontend UI (Flask)                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Starting on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py
