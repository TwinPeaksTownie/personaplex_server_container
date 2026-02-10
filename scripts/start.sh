#!/bin/bash
set -e

# Download and extract built-in voice prompts from HuggingFace if not already present
if [ ! -f /app/voices/NATF0.pt ]; then
    echo "Downloading built-in voice prompts from HuggingFace..."
    /app/moshi/.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import tarfile, os
tgz = hf_hub_download('nvidia/personaplex-7b-v1', 'voices.tgz')
with tarfile.open(tgz, 'r:gz') as tar:
    tar.extractall(path='/tmp/hf_voices')
# Symlink .pt files into /app/voices/ (don't overwrite existing custom voices)
for f in os.listdir('/tmp/hf_voices/voices'):
    src = os.path.join('/tmp/hf_voices/voices', f)
    dst = os.path.join('/app/voices', f)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f'  linked {f}')
"
    echo "Voice prompts ready."
fi

# Start Python backend on loopback only
cd /app/moshi
/app/moshi/.venv/bin/python -m moshi.server \
    --host 127.0.0.1 \
    --port 8080 \
    --static none \
    --voice-prompt-dir /app/voices &
BACKEND_PID=$!

# Wait for backend health
echo "Waiting for Moshi backend to initialize..."
until curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "ERROR: Backend process died during startup"
        exit 1
    fi
    sleep 2
done
echo "Moshi backend is ready."

# Start Nginx gateway
echo "Starting Nginx gateway on port 5173..."
nginx -g 'daemon off;' &
FRONTEND_PID=$!

# If either process dies, kill the other and exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 1" SIGTERM SIGINT
wait -n $BACKEND_PID $FRONTEND_PID
echo "ERROR: A process exited unexpectedly"
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
exit 1
