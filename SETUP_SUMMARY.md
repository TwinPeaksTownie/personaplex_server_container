# PersonaPlex Server Container - Setup Summary

## ✅ What's Been Created

Your repository is now ready for both **Brev deployment** AND **deep source code analysis**!

### Repository Structure

```
i:\AI_stuff\personaplex_server_container/
├── 📄 README.md                      # Main documentation with quick start
├── 🐳 docker-compose.yml             # Production multi-service setup
├── 🐳 Dockerfile.backend             # Backend container config
├── 📝 .env.example                   # Environment template
├── 🚫 .gitignore                     # Excludes sensitive/large files
├── 🚫 .dockerignore                  # Optimizes build context
│
├── 📁 moshi/                         # FULL BACKEND SOURCE CODE
│   ├── moshi/
│   │   ├── server.py                # WebSocket server (START HERE)
│   │   ├── models.py                # Model inference logic
│   │   ├── audio.py                 # Opus codec handling
│   │   ├── prompt.py                # Voice/text prompts
│   │   └── ...                      # Complete source
│   └── pyproject.toml               # Python dependencies
│
├── 📁 client/                        # FULL FRONTEND SOURCE CODE
│   ├── src/                         # React/TypeScript app
│   ├── Dockerfile                   # Frontend container
│   └── package.json                 # Node dependencies
│
├── 📁 voices/                        # Voice prompt directory
│   └── .gitkeep                     # Placeholder
│
└── 📁 docs/
    ├── BREV_DEPLOYMENT.md           # Cloud deployment guide
    └── LOCAL_DEVELOPMENT.md         # Local dev guide
```

---

## 🎯 What You Can Do Now

### 1. **Push to GitHub**

```bash
cd i:\AI_stuff\personaplex_server_container

git commit -m "Initial setup: Full PersonaPlex source + Docker configs"
git push -u origin main
```

### 2. **Run DeepWiki Analysis**

Once pushed to GitHub:

```bash
# Run DeepWiki to generate comprehensive documentation
deepwiki analyze https://github.com/TwinPeaksTownie/personaplex_server_container
```

**Key files to analyze:**
- `moshi/moshi/server.py` - WebSocket protocol
- `moshi/moshi/models.py` - Model architecture
- `client/src/` - Frontend implementation

### 3. **Deploy on Brev**

1. Go to [brev.dev](https://brev.dev)
2. Create new Launchable
3. Point to: `https://github.com/TwinPeaksTownie/personaplex_server_container`
4. Select Docker Compose runtime
5. Add `HF_TOKEN` environment variable
6. Choose GPU (1x T4 minimum, 1x A40 recommended)
7. Launch!

### 4. **Test Locally**

```bash
cd i:\AI_stuff\personaplex_server_container

# Configure
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Run
docker-compose up --build
```

Access at:
- Frontend: http://localhost:5173
- Backend: http://localhost:8080

---

## 🧠 Understanding the Backend

### Architecture Overview

```
┌─────────────┐     WebSocket      ┌──────────────┐
│   Client    │ ←─────────────────→ │ Moshi Server │
│  (Browser)  │   Opus Audio       │  (FastAPI)   │
└─────────────┘   + Metadata       └──────┬───────┘
                                          │
                                          ↓
                                   ┌──────────────┐
                                   │ PersonaPlex  │
                                   │   Model      │
                                   │  (7B params) │
                                   └──────────────┘
```

### Key Entry Points for Analysis

1. **WebSocket Handler**
   - File: `moshi/moshi/server.py`
   - Function: `handle_websocket()`
   - Purpose: Real-time audio streaming protocol

2. **Model Inference**
   - File: `moshi/moshi/models.py`
   - Class: `PersonaPlexModel`
   - Purpose: Voice conditioning + text generation

3. **Audio Processing**
   - File: `moshi/moshi/audio.py`
   - Class: `OpusEncoder`/`OpusDecoder`
   - Purpose: 24kHz Opus codec handling

4. **Prompt Management**
   - File: `moshi/moshi/prompt.py`
   - Purpose: Voice embeddings + text prompts

### Data Flow

```
User speaks → Browser captures audio → Opus encode (24kHz mono)
    ↓
WebSocket send → Server receives → Opus decode
    ↓
Model processes (voice + text conditioning) → Generate response
    ↓
Opus encode response → WebSocket send → Browser plays
```

---

## 📊 What's Different from NVIDIA's Repo

| Aspect | NVIDIA Original | Your Fork |
|--------|----------------|-----------|
| **Docker Setup** | Basic single service | Production multi-service |
| **Frontend** | Separate port 8998 | Proxied through 5173 |
| **Backend** | SSL required | Flexible (SSL optional) |
| **Deployment** | Manual setup | Brev-ready Launchable |
| **Documentation** | Basic README | Full guides + architecture |
| **Voice Prompts** | Embedded in code | Volume-mounted directory |
| **Environment** | Hardcoded | `.env` file support |

---

## 🔍 DeepWiki Analysis Targets

When you run DeepWiki, focus on:

### Backend Deep Dive
- [ ] `moshi/moshi/server.py` - WebSocket protocol
- [ ] `moshi/moshi/models.py` - Model architecture
- [ ] `moshi/moshi/audio.py` - Audio codec
- [ ] `moshi/moshi/prompt.py` - Prompt system

### Frontend Deep Dive
- [ ] `client/src/App.tsx` - Main application
- [ ] `client/src/components/` - UI components
- [ ] `client/src/hooks/` - WebSocket hooks

### Configuration
- [ ] `moshi/pyproject.toml` - Python dependencies
- [ ] `client/package.json` - Node dependencies
- [ ] `docker-compose.yml` - Service orchestration

---

## 🚀 Next Steps

1. **Commit and push** to GitHub
2. **Run DeepWiki** for comprehensive analysis
3. **Create Brev Launchable** for cloud deployment
4. **Test locally** with Docker Compose
5. **Customize** voice prompts and personas

---

## 📝 Git Commands Reference

```bash
# Check status
git status

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push origin main

# Pull latest (if collaborating)
git pull origin main
```

---

## 🎤 Voice Prompt Setup

To add custom voices:

1. Place `.wav` files in `voices/` directory
2. Restart backend: `docker-compose restart personaplex-backend`
3. Select from frontend dropdown

**Example:**
```bash
voices/
├── Laura.wav       # Your custom voice
├── Alex.wav        # Another voice
└── .gitkeep        # Git placeholder
```

---

## 🐛 Troubleshooting

### "HF_TOKEN not set"
- Copy `.env.example` to `.env`
- Add your token from https://huggingface.co/settings/tokens

### "Port already in use"
- Change ports in `docker-compose.yml`
- Or stop conflicting services

### "CUDA out of memory"
- Use GPU with more VRAM
- Enable CPU offload in `docker-compose.yml`

---

## 📚 Resources

- **Brev Deployment**: See `docs/BREV_DEPLOYMENT.md`
- **Local Development**: See `docs/LOCAL_DEVELOPMENT.md`
- **API Reference**: See `API_REFERENCE.md` (from original repo)
- **PersonaPlex Paper**: https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf

---

## ✨ Summary

You now have:
- ✅ **Full PersonaPlex source code** for deep analysis
- ✅ **Production Docker setup** for easy deployment
- ✅ **Brev-ready configuration** for cloud GPU access
- ✅ **Comprehensive documentation** for both use cases
- ✅ **Git repository** ready to push to GitHub

**The best of both worlds!** 🎉
