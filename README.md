# PersonaPlex Server Container

[![Deploy on Brev](https://img.shields.io/badge/Deploy%20on-Brev-blue)](https://brev.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-GPU%20Required-76B900?logo=nvidia)](https://www.nvidia.com/)

**Full-duplex conversational AI with voice and persona control** - containerized for easy deployment on NVIDIA GPUs.

This repository contains the complete PersonaPlex source code plus production-ready Docker configurations for:
- 🚀 **One-click Brev deployment** with GPU support
- 🐳 **Local Docker Compose** setup
- 📚 **Full source code** for deep analysis and customization
- 🔧 **Unified Gateway Architecture** (Frontend + Backend in one container)
- 🔒 **SSL/TLS enabled** for secure WebSocket handshakes

---

## 🎯 Quick Start Options

### Option 1: Deploy on Brev (Recommended for Cloud)

1. Click the "Deploy on Brev" button above
2. Select GPU: **1x NVIDIA T4** (minimum) or **1x A40** (recommended)
3. Add your `HF_TOKEN` environment variable
4. Launch! 🎉

### Option 2: Local Docker Compose

**Prerequisites:**
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with 16GB+ VRAM
- Hugging Face account with PersonaPlex license accepted

**Setup:**
```bash
# Clone this repo
git clone https://github.com/TwinPeaksTownie/personaplex_server_container.git
cd personaplex_server_container

# Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Launch
docker-compose up --build
```

**Access:**
- **Web UI:** https://localhost:5173 (Self-signed cert)
- **Backend API:** https://localhost:5173/api (Proxied)

---

## 📁 Repository Structure

```
personaplex_server_container/
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile.backend          # Backend container (Moshi server)
├── .env.example                # Environment template
│
├── moshi/                      # Backend source code
│   ├── moshi/                  # Python package
│   │   ├── server.py          # WebSocket server
│   │   ├── models.py          # Model loading & inference
│   │   └── ...
│   └── pyproject.toml         # Python dependencies
│
├── client/                     # Frontend source code
│   ├── src/                   # React/TypeScript app
│   ├── Dockerfile             # Frontend container
│   └── package.json           # Node dependencies
│
├── voices/                     # Voice prompt embeddings (optional)
└── docs/                       # Additional documentation
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | ✅ Yes | Hugging Face token ([get one](https://huggingface.co/settings/tokens)) |
| `VITE_QUEUE_API_URL` | No | Backend URL (default: `http://personaplex-backend:8080`) |
| `NO_TORCH_COMPILE` | No | Set to `1` for faster startup (recommended) |

### GPU Requirements

| GPU | VRAM | Performance | Use Case |
|-----|------|-------------|----------|
| NVIDIA T4 | 16GB | Good (8-bit) | Development, demos |
| NVIDIA A40 | 48GB | Excellent | Production, full precision |
| NVIDIA A100 | 80GB | Best | High-load production |

---

## 🧠 Understanding the Backend

This repo includes the **complete PersonaPlex source code** for deep analysis:

### Key Backend Components

1. **`moshi/moshi/server.py`** - WebSocket server handling real-time audio streaming
2. **`moshi/moshi/models.py`** - Model loading, inference, and voice conditioning
3. **`moshi/moshi/audio.py`** - Opus codec handling and audio processing
4. **`moshi/moshi/prompt.py`** - Text and voice prompt management

### Architecture Overview

```
┌─────────────┐     HTTPS/WSS      ┌──────────────────┐
│   Client    │ ←─────────────────→ │  Unified Gateway │
│  (Browser)  │   Opus Audio       │ (Vite + Moshi)   │
└─────────────┘   + Metadata       └──────────────────┘
```

**Recommended for Deep Dive:**
- Run [DeepWiki](https://github.com/deepwiki/deepwiki) on this repo to generate comprehensive documentation
- Start with `moshi/moshi/server.py` to understand the WebSocket protocol
- Explore `moshi/moshi/models.py` for model inference details

---

## 🎤 Voice Prompts

PersonaPlex supports custom voice embeddings. Place `.pt` files in the `voices/` directory:

```bash
voices/
├── Laura.wav       # Your custom voice sample
└── NATF2.pt        # Pre-generated embedding
```

**Pre-packaged voices:**
- Natural (female): `NATF0`, `NATF1`, `NATF2`, `NATF3`
- Natural (male): `NATM0`, `NATM1`, `NATM2`, `NATM3`
- Variety (female): `VARF0-4`
- Variety (male): `VARM0-4`

---

## 🐛 Troubleshooting

### Container won't start
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi

# Check logs
docker-compose logs personaplex-backend
```

### Frontend can't connect to backend
- Ensure `VITE_QUEUE_API_URL` uses the service name `personaplex-backend` (not `localhost`)
- Check health status: `curl http://localhost:8080/health`

### Out of memory errors
- Use a GPU with more VRAM
- Enable CPU offload: add `--cpu-offload` to the backend command in `docker-compose.yml`

---

## 📚 Additional Resources

- [Official PersonaPlex Paper](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
- [Model Weights](https://huggingface.co/nvidia/personaplex-7b-v1)
- [Moshi Architecture](https://arxiv.org/abs/2410.00037)
- [FullDuplexBench Evaluation](https://arxiv.org/abs/2503.04721)

---

## 📝 License

- **Code**: MIT License (see `LICENSE-MIT`)
- **Model Weights**: NVIDIA Open Model License

---

## 🤝 Contributing

This is a containerized deployment repo. For core PersonaPlex contributions, see the [official NVIDIA repository](https://github.com/NVIDIA/personaplex).

For container/deployment improvements:
1. Fork this repo
2. Create a feature branch
3. Test with `docker-compose up --build`
4. Submit a PR

---

## 🙏 Acknowledgments

Built on [PersonaPlex](https://github.com/NVIDIA/personaplex) by NVIDIA Research.
