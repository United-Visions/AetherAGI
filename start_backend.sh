#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           AetherMind Backend API (FastAPI)                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Starting on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
uvicorn orchestrator.main_api:app --reload --host 0.0.0.0 --port 8000
