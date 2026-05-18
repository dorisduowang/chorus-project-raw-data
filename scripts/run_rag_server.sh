#!/bin/bash
# CHORUS RAG Server Launcher
#
# Usage:
#   ./run_rag_server.sh              # Development mode (auto-reload)
#   ./run_rag_server.sh production   # Production mode (4 workers)
#   ./run_rag_server.sh legacy       # Legacy http.server mode

set -e

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default settings
export RAG_PORT=${RAG_PORT:-8765}
MODE=${1:-development}

# Activate virtual environment if exists
if [ -d "venv_rag" ]; then
    source venv_rag/bin/activate
    echo "Using venv_rag virtual environment"
fi

echo "========================================"
echo "CHORUS RAG Server"
echo "========================================"
echo "Mode: $MODE"
echo "Port: $RAG_PORT"
echo ""

case $MODE in
    production)
        export CHORUS_ENV=production
        echo "Starting FastAPI server (4 workers)..."
        python rag_server_fastapi.py
        ;;
    development|dev)
        export CHORUS_ENV=development
        echo "Starting FastAPI server (auto-reload)..."
        python rag_server_fastapi.py
        ;;
    legacy|old)
        echo "Starting legacy http.server..."
        python rag_http_server.py
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [development|production|legacy]"
        exit 1
        ;;
esac
