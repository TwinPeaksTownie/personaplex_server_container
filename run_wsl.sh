#!/bin/bash
# PersonaPlex Launch Script - Native WSL Performance

# --- AUTOMATIC CLEANUP ---
echo "🧹 Cleaning up old PersonaPlex processes..."
pkill -9 -f moshi.server > /dev/null 2>&1
pkill -9 -f vite > /dev/null 2>&1
pkill -9 -f node > /dev/null 2>&1
# -------------------------

cd /home/carson/personaplex

# Configuration
export HF_HOME=/home/carson/.cache/huggingface
mkdir -p "$HF_HOME"

# Backend Setup
echo "🚀 Starting PersonaPlex Backend (Moshi)..."
source moshi/.venv/bin/activate

# Start Backend
cd moshi
# Port 8080 is default for PersonaPlex API.
python3 -m moshi.server --voice-prompt-dir voices --host 0.0.0.0 --port 8080 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Frontend Setup
echo "🎨 Starting PersonaPlex Frontend (Vite)..."
cd client
export VITE_QUEUE_API_URL=http://localhost:8080
npm run dev -- --host --port 5173 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "✅ PersonaPlex is starting!"
echo "🔓 Note: Uses HTTP (No SSL)."
echo "📱 Local (WSL): http://localhost:5173"
echo "🌐 Network: http://$(hostname -I | awk '{print $1}'):5173"
echo ""
echo "📝 Log files: backend.log, frontend.log"
echo "🛑 To stop: kill $BACKEND_PID $FRONTEND_PID"

# Keep script running to monitor
wait $BACKEND_PID $FRONTEND_PID
