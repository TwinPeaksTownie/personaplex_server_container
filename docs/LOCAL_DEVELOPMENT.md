# Local Development Setup

## Prerequisites

- Docker Desktop with WSL2 backend (Windows) or Docker Engine (Linux/Mac)
- NVIDIA Container Toolkit
- NVIDIA GPU with 16GB+ VRAM
- Git

---

## Installation

### 1. Install NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Windows (WSL2):**
```powershell
# Install Docker Desktop with WSL2 backend
# NVIDIA Container Toolkit is included automatically
```

### 2. Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

You should see your GPU listed.

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/TwinPeaksTownie/personaplex_server_container.git
cd personaplex_server_container
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:
```bash
HF_TOKEN=hf_your_token_here
```

### 3. Build and Run

```bash
docker-compose up --build
```

**First run will:**
- Download CUDA base image (~2GB)
- Install dependencies
- Download PersonaPlex model (~14GB)
- Download built-in voice prompts
- Start backend + Nginx gateway

**Expect 10-15 minutes for first build.**

---

## Accessing the Application

Once running:
- **Web UI**: https://localhost:5173 (accept the self-signed cert warning)
- **API (proxied)**: https://localhost:5173/api
- **Health Check**: https://localhost:5173/health

> **Note:** The backend runs on `127.0.0.1:8080` inside the container and is not exposed externally. All access goes through the Nginx gateway on port 5173.

---

## Development Workflow

### Backend Changes

After modifying backend code, rebuild the container:
```bash
docker-compose up --build
```

Or restart the running container:
```bash
docker-compose restart personaplex
```

### View Logs

```bash
# All output (backend + nginx)
docker-compose logs -f

# Follow logs for the unified container
docker-compose logs -f personaplex
```

---

## Debugging

### Enter the Container

```bash
docker-compose exec personaplex bash

# Inside container:
/app/moshi/.venv/bin/python -m moshi.server --help
```

### Check GPU Usage

```bash
# From host
nvidia-smi

# From inside the container
docker-compose exec personaplex nvidia-smi
```

---

## Custom Voice Prompts

### Add Voice Files

1. Place `.wav` files in `voices/` directory:
   ```bash
   voices/
   └── MyVoice.wav
   ```

2. Restart the container:
   ```bash
   docker-compose restart personaplex
   ```

3. The voice will appear in the frontend dropdown (fetched dynamically from `/api/voices`)

---

## Performance Tuning

### Enable CPU Offload

For GPUs with <24GB VRAM, edit `scripts/start.sh` and add `--cpu-offload` to the python command:

```bash
/app/moshi/.venv/bin/python -m moshi.server \
    --host 127.0.0.1 \
    --port 8080 \
    --static none \
    --voice-prompt-dir /app/voices \
    --cpu-offload &
```

Then rebuild: `docker-compose up --build`

### Disable Torch Compilation

Already set in `.env`:
```bash
NO_TORCH_COMPILE=1
```

This speeds up startup by ~2 minutes.

---

## Split Mode (Two Containers)

For debugging backend and frontend separately, use the split compose file:

```bash
docker-compose -f docker-compose.split.yml up --build
```

This runs:
- `personaplex-backend` on port 8080 (backend only)
- `personaplex-frontend` on port 5173 (Vite dev server)

---

## Stopping and Cleanup

### Stop Containers

```bash
docker-compose down
```

### Remove Volumes (Fresh Start)

```bash
docker-compose down -v
```

**Warning**: This deletes the downloaded model cache (~14GB). Next start will re-download.

### Clean Docker System

```bash
docker system prune -a
```

---

## Troubleshooting

### Port Already in Use

**Error**: `Bind for 0.0.0.0:5173 failed: port is already allocated`

**Solution**:
```bash
# Find process using port
netstat -ano | findstr :5173  # Windows
lsof -i :5173                 # Linux/Mac

# Kill process or change port in docker-compose.yml
```

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Close other GPU applications
2. Enable CPU offload (see Performance Tuning)
3. Use a GPU with more VRAM

### WebSocket Connection Fails

**Solution**:
```bash
# Check backend health through the gateway
curl -k https://localhost:5173/health

# Check container logs for errors
docker-compose logs personaplex | grep -i error
```

---

## Advanced: Source Code Exploration

### Backend Architecture

```
moshi/
├── moshi/
│   ├── server.py              # aiohttp server (WebSocket chat, voices, MCP, health)
│   ├── models/
│   │   ├── loaders.py         # Model downloading & initialization
│   │   ├── lm.py              # Language model (LMModel, LMGen)
│   │   └── compression.py     # Mimi audio codec
│   ├── modules/               # Neural network building blocks (transformer, seanet, etc.)
│   ├── quantization/          # Vector quantization (VQ)
│   ├── utils/                 # Utilities (connection, compile, autocast, logging, sampling)
│   ├── client_utils.py        # Client utilities
│   └── offline.py             # Offline inference mode
└── pyproject.toml             # Dependencies
```

### Key Entry Points

1. **WebSocket Handler**: `moshi/moshi/server.py` → `ServerState.handle_chat()`
2. **Model Inference**: `moshi/moshi/models/lm.py` → `LMGen.step()`
3. **Audio Codec**: `moshi/moshi/models/compression.py` → Mimi model (encode/decode)
4. **Opus Streaming**: Uses `sphn` library (`OpusStreamReader` / `OpusStreamWriter`)

### Running Tests

```bash
docker-compose exec personaplex bash -c "cd /app/moshi && /app/moshi/.venv/bin/python -m pytest"
```

---

## Next Steps

- Read [BREV_DEPLOYMENT.md](BREV_DEPLOYMENT.md) for cloud deployment
- See [UNIFIED_GATEWAY_BUILD.md](UNIFIED_GATEWAY_BUILD.md) for full API reference and binary protocol
- See [SERVER_UPDATE_REBUILD.md](SERVER_UPDATE_REBUILD.md) for MCP endpoint documentation
