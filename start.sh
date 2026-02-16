#!/bin/bash
set -e

# Start the PersonaPlex Backend + Frontend (Unified Gateway)
# The UI is served as static content via the Python server
echo "Starting Unified PersonaPlex Gateway on Port 8080..."

/app/moshi/.venv/bin/python -m moshi.server \
    --voice-prompt-dir /app/voices \
    --host 0.0.0.0 \
    --port 8080 \
    --static /app/client/dist

# Wait for process to exit
wait
