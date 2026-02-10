# Unified Gateway Architecture: Technical Implementation Plan

This plan refactors the PersonaPlex project into a **Unified Gateway Architecture**. This ensures the **Robot's Internal Browser** can correctly perform the mandatory SSL Handshake and Binary Protocol negotiation through the Vite Dev Server (Port 5173).

## 1. The Core Infrastructure (Unified Dockerfile)
We merge the Node.js and Python environments into a single image to eliminate cross-container networking latency and certificate mismatch issues.

```dockerfile
# Base: NVIDIA CUDA Runtime for GPU support
FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System Dependencies
RUN apt-get update && apt-get install -y \
    curl git nodejs npm python3.12 python3.12-venv openssl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast Python management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Step A: Build the Vite Client Gateway
COPY ./client /app/client
RUN cd /app/client && npm ci

# Generate SSL Certificates for the Gateway
RUN cd /app/client && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem -out cert.pem \
    -subj "/C=US/ST=NC/L=Durham/O=NVIDIA/OU=PersonaPlex/CN=localhost"

# Step B: Build the Moshi Python Backend
COPY . /app/server
RUN cd /app/server && uv sync

# Step C: Orchestration Script
COPY scripts/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 5173 8080
ENTRYPOINT ["/app/start.sh"]
```

## 2. Gateway Configuration (`vite.config.ts`)
The Vite server on Port 5173 acts as the **only** public entrance. It handles the SSL/WSS termination and proxies traffic to the local Python process.

```typescript
export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5173,
    https: {
      key: fs.readFileSync('./key.pem'),
      cert: fs.readFileSync('./cert.pem'),
    },
    proxy: {
      // Proxies Browser Handshake/WebSocket to the Python Brain
      '/api': {
        target: 'http://127.0.0.1:8080',
        ws: true,
        changeOrigin: true,
        secure: false // Allow self-signed certs
      }
    }
  }
});
```

## 3. Orchestration Logic (`start.sh`)
Ensures the Backend is ready before the Gateway starts accepting robot connections.

```bash
#!/bin/bash
# 1. Start Python Moshi Engine on the internal loopback
cd /app/server && uv run python -m moshi.server --host 127.0.0.1 --port 8080 --voice-prompt-dir /app/voices &

# 2. Local Health Check (Wait for the 7B Model to load in VRAM)
until curl -s http://127.0.0.1:8080/health; do
  echo "Waiting for Moshi Brain to initialize..."
  sleep 5
done

# 3. Start Vite Gateway (Handshake + UI Provider)
cd /app/client && npm run dev
```

## 4. Reachy Mac App Lifecycle (Plan B)
How the handshake succeeds in this refactored model:

1.  **Identity Initialization**: The Reachy App (Mac) spawns its browser pointing to `https://<PC_IP>:5173`.
2.  **SSL Handshake**: The browser verifies the `cert.pem`. Because the gateway and backend are now unified, there is no "Mixed Content" error.
3.  **The "Gap" (UI State)**: The user/app picks the Persona in the browser.
4.  **Binary Protocol Handshake**:
    *   Browser initiates `WSS://<PC_IP>:5173/api/chat`.
    *   Vite proxies this to Python (`8080`).
    *   Python sends byte `0x00`.
    *   Browser (using `encoder.ts`) responds with `[0x00, 0x00, 0x00]`.
5.  **Steady State**: Audio streaming begins. The robot's USB hardware is now connected to the AI logic via the established 5173 tunnel.

## 5. Summary of Improvements
*   **Eliminates Handshake Timeouts**: The proxy bridge is internal to the container.
*   **Self-Contained SSL**: Certificates are managed once at the gateway.
*   **Production Port**: Only one port (5173) needs to be exposed for the full robot experience.
