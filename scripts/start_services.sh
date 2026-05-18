#!/bin/bash
# Start both Jupyter and RAG server in the same container

# Start RAG server in background
echo "Starting RAG FastAPI server on port 8765..."
python /app/rag_server_fastapi.py &
RAG_PID=$!

# Wait a moment for RAG server to initialize
sleep 3

# Check if RAG server started
if kill -0 $RAG_PID 2>/dev/null; then
    echo "RAG server started (PID: $RAG_PID)"
else
    echo "WARNING: RAG server failed to start"
fi

# Start Jupyter (foreground - keeps container running)
echo "Starting Jupyter notebook on port 8888..."
exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
