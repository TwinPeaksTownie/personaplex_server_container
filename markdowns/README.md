# PersonaPlex: Full-Duplex AI Agent

PersonaPlex is a low-latency, real-time voice AI built on the Moshi architecture. It is optimized for local deployment on NVIDIA GPUs for use with robotics (Reachy Mini) and native clients.

## 🚀 Quick Start

### 1. Requirements
- **Hardware**: NVIDIA GPU (12GB+ VRAM recommended)
- **Software**: Docker + NVIDIA Container Toolkit

### 2. Launch
```powershell
# Start the unified container (Port 8080 Backend, 5173 Frontend)
docker-compose up -d --build
```

### 3. Connect
- **Browser**: `https://localhost:5173`
- **Native Client**: `wss://[IP]:8080/api/chat`

## 📂 Project Structure
- `moshi/`: Python backend engine logic.
- `client/`: React/Vite web interface.
- `voices/`: Voice prompt storage (drop `.wav` files here).
- `ssl/`: SSL Certificates for secure WSS streaming.

## 📖 Internal Documentation
- **[HANDOFF.md](./HANDOFF.md)**: **The Source of Truth.** Binary protocols, handshake sequence, and API specs.
- **[REACHY_INTEGRATION.md](./REACHY_INTEGRATION.md)**: Robot-specific SDK examples for Reachy Mini hardware.
