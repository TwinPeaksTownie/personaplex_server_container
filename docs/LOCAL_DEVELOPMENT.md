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
- Start backend and frontend

**Expect 10-15 minutes for first build.**

---

## Accessing the Application

Once running:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8080
- **Health Check**: http://localhost:8080/health

---

## Development Workflow

### Hot Reload (Frontend)

Frontend changes auto-reload. Just edit files in `client/src/`.

### Backend Changes

After modifying backend code:
```bash
docker-compose restart personaplex-backend
```

For dependency changes:
```bash
docker-compose up --build personaplex-backend
```

### View Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f personaplex-backend

# Frontend only
docker-compose logs -f personaplex-frontend
```

---

## Debugging

### Enter Backend Container

```bash
docker-compose exec personaplex-backend bash

# Inside container:
python -m moshi.server --help
```

### Enter Frontend Container

```bash
docker-compose exec personaplex-frontend sh

# Inside container:
npm run build
```

### Check GPU Usage

```bash
# From host
nvidia-smi

# From container
docker-compose exec personaplex-backend nvidia-smi
```

---

## Custom Voice Prompts

### Add Voice Files

1. Place `.wav` files in `voices/` directory:
   ```bash
   voices/
   └── MyVoice.wav
   ```

2. Restart backend:
   ```bash
   docker-compose restart personaplex-backend
   ```

3. Use in frontend by selecting "MyVoice" from dropdown

---

## Performance Tuning

### Enable CPU Offload

For GPUs with <24GB VRAM, edit `docker-compose.yml`:

```yaml
services:
  personaplex-backend:
    command: [
      "/app/moshi/.venv/bin/python", "-m", "moshi.server",
      "--voice-prompt-dir", "/app/voices",
      "--host", "0.0.0.0",
      "--port", "8080",
      "--cpu-offload"  # Add this line
    ]
```

### Disable Torch Compilation

Already set in `.env`:
```bash
NO_TORCH_COMPILE=1
```

This speeds up startup by ~2 minutes.

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

**Error**: `Bind for 0.0.0.0:8080 failed: port is already allocated`

**Solution**:
```bash
# Find process using port
netstat -ano | findstr :8080  # Windows
lsof -i :8080                 # Linux/Mac

# Kill process or change port in docker-compose.yml
```

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Close other GPU applications
2. Enable CPU offload (see Performance Tuning)
3. Use a GPU with more VRAM

### Frontend Can't Connect

**Error**: `WebSocket connection failed`

**Solution**:
```bash
# Check backend is running
curl http://localhost:8080/health

# Check VITE_QUEUE_API_URL in .env
# Should be: http://personaplex-backend:8080
```

---

## Advanced: Source Code Exploration

### Backend Architecture

```
moshi/
├── moshi/
│   ├── server.py          # FastAPI WebSocket server
│   ├── models.py          # Model loading & inference
│   ├── audio.py           # Opus codec handling
│   ├── prompt.py          # Voice/text prompt management
│   └── utils.py           # Utilities
└── pyproject.toml         # Dependencies
```

### Key Entry Points

1. **WebSocket Handler**: `moshi/moshi/server.py:handle_websocket()`
2. **Model Inference**: `moshi/moshi/models.py:PersonaPlexModel.generate()`
3. **Audio Processing**: `moshi/moshi/audio.py:OpusEncoder`

### Running Tests

```bash
docker-compose exec personaplex-backend bash -c "cd /app/moshi && python -m pytest"
```

---

## Next Steps

- Read [BREV_DEPLOYMENT.md](BREV_DEPLOYMENT.md) for cloud deployment
- Explore [API_REFERENCE.md](../API_REFERENCE.md) for WebSocket protocol
- Run DeepWiki on this repo for comprehensive documentation
