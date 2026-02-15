# PersonaPlex Unified Gateway: Build & Handoff Guide

## What This Is

A single Docker container that runs the PersonaPlex 7B voice AI model behind an Nginx HTTPS gateway. One exposed port (5173), one container, GPU-accelerated inference with per-connection voice and persona swapping.

## Prerequisites

- Docker with NVIDIA Container Toolkit (`--gpus all` support)
- GPU with ~16GB VRAM (RTX 4090, A5000, L40, etc.)
- A HuggingFace token with access to [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)

## Quick Start

1. Create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```

2. Build and run:
```bash
docker compose build
docker compose up
```

3. Wait for logs to show:
```
Moshi backend is ready.
Starting Nginx gateway on port 5173...
```

4. Open `https://<host-ip>:5173` in a browser (accept the self-signed cert).

First startup downloads ~14GB of model weights from HuggingFace. This is cached in a Docker volume (`huggingface-cache`) and skipped on subsequent runs.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Container                               │
│                                                 │
│  ┌─────────────────────┐                        │
│  │  Nginx Gateway      │  ← HTTPS :5173         │
│  │  (static React UI)  │     (only exposed port) │
│  │                     │                        │
│  │  /api/* proxy ──────┼──► 127.0.0.1:8080      │
│  │  (WebSocket)        │                        │
│  └─────────────────────┘                        │
│                                                 │
│  ┌─────────────────────┐                        │
│  │  Moshi Server       │  ← HTTP :8080          │
│  │  (Python/PyTorch)   │     (loopback only)    │
│  │                     │                        │
│  │  PersonaPlex 7B     │  ← loaded in VRAM      │
│  │  Mimi Audio Codec   │                        │
│  └─────────────────────┘                        │
│                                                 │
│  Volumes:                                       │
│    /app/voices ← ./voices (host mount)          │
│    /app/.cache ← huggingface-cache (named vol)  │
└─────────────────────────────────────────────────┘
```

---

## API Endpoints

All endpoints are accessed through the Nginx gateway on port 5173.

### `GET /health`
Health check. Returns `{"status": "ok"}` when the model is loaded.

### `GET /api/voices`
Lists available voice files (`.wav` and `.pt`) from the voices directory.

```json
{"voices": ["NATF0.pt", "NATM1.pt", "custom-voice.wav"]}
```

### `POST /api/voices/upload`
Upload a `.wav` voice file. Multipart form data, max 50MB.

```json
{"status": "ok", "filename": "my-voice.wav", "size": 123456}
```

### `GET /api/chat` (WebSocket)
Main voice streaming endpoint. Upgrades to WebSocket.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice_prompt` | string | required | Voice file name (e.g., `NATF0.pt` or `custom.wav`) |
| `text_prompt` | string | `""` | System instructions for the AI persona |
| `text_temperature` | float | 0.7 | Text generation temperature |
| `text_topk` | int | 25 | Text top-k sampling |
| `audio_temperature` | float | 0.8 | Audio generation temperature |
| `audio_topk` | int | 250 | Audio top-k sampling |
| `repetition_penalty` | float | 1.0 | Repetition penalty |
| `repetition_penalty_context` | int | 64 | Context window for repetition penalty |
| `pad_mult` | float | 0 | Padding multiplier |
| `seed` | int | -1 | Random seed (-1 = random) |

**Binary Protocol:**

| Direction | Byte Prefix | Payload | Meaning |
|-----------|-------------|---------|---------|
| Server → Client | `\x00` | none | Handshake complete, ready for audio |
| Client → Server | `\x01` | Opus audio | User speech |
| Server → Client | `\x01` | Opus audio | AI speech |
| Server → Client | `\x02` | UTF-8 text | Live transcription |

**Connection lifecycle:**
1. Client opens WebSocket with query params
2. Server loads voice prompt (from `.pt` cache or encodes `.wav`)
3. Server processes text system prompt
4. Server sends `\x00` handshake byte
5. Bidirectional audio streaming begins
6. Either side closes the WebSocket to end the session

---

## Voice Prompt System

### Built-in Voices (downloaded from HuggingFace)
- `NATF0.pt` through `NATF3.pt` — Natural female voices
- `NATM0.pt` through `NATM3.pt` — Natural male voices
- `VARF0.pt` through `VARF4.pt` — Variety female voices
- `VARM0.pt` through `VARM4.pt` — Variety male voices

### Custom Voices
Drop a `.wav` file into the `./voices/` directory (host-mounted). On first use, the server:
1. Loads the audio, normalizes to -24 LUFS
2. Encodes through the Mimi audio codec
3. Runs each frame through the 7B transformer
4. Saves the resulting embeddings as a `.pt` file alongside the `.wav`

Subsequent connections using the same voice load the `.pt` directly, skipping the expensive encoding. This is controlled by `save_voice_prompt_embeddings=True` in `moshi/moshi/server.py:509`.

### Audio Requirements
- Format: WAV
- Any sample rate (resampled to 32kHz internally)
- Mono preferred (stereo is handled)
- No minimum/maximum duration enforced
- Longer clips = more transformer steps on first use = better voice quality but slower first connection

---

## System Prompts (Personas)

The `text_prompt` query parameter accepts any string. It's wrapped in `<system>` tags internally and tokenized as text input to the model. Examples:

```
You are a wise and friendly teacher.
You work for Dr. Jones's medical office. Help patients schedule appointments.
You are a pirate named Captain Byte. Speak like a salty sea dog.
```

System prompts are processed per-connection (cheap — just token stepping, no audio encoding). Changing the system prompt between connections is effectively instant.

---

## Per-Connection Persona Swapping

The model stays loaded in VRAM at all times. Only voice embeddings and text prompts change between sessions:

| Change | Cost | What Happens |
|--------|------|-------------|
| Same voice, same prompt | ~0.5s | Replay cached embeddings + silence padding |
| Same voice, new prompt | ~0.5-1s | Replay cached embeddings + new text tokens |
| New voice (`.pt` exists) | ~0.5-1s | Load `.pt` from disk + text tokens |
| New voice (`.wav`, first use) | ~2-10s | Encode audio through Mimi + transformer, save `.pt` |

---

## File Structure

```
personaplex_server_container/
├── Dockerfile.unified          # Unified container build
├── docker-compose.yml          # Single-service compose
├── scripts/
│   └── start.sh                # Process orchestration
├── .env                        # HF_TOKEN goes here
├── voices/                     # Voice prompt files (host-mounted)
├── client/                     # Vite + React frontend
│   ├── vite.config.ts          # Gateway config (SSL, proxy)
│   ├── src/
│   │   ├── app.tsx             # App entry point
│   │   └── pages/
│   │       ├── Queue/          # Persona selection UI
│   │       └── Conversation/   # WebSocket audio streaming
│   └── package.json
└── moshi/                      # Python backend
    ├── pyproject.toml
    └── moshi/
        ├── server.py           # HTTP/WS server, API routes
        └── models/
            ├── loaders.py      # Model download + initialization
            └── lm.py           # LM inference, voice prompt caching
```

---

## Key Configuration Files

### `nginx.conf`
- SSL termination with self-signed cert (generated at build time)
- Proxies `/api/*` → `http://127.0.0.1:8080` with WebSocket upgrade support
- Proxies `/mcp` and `/health` → backend
- Serves static React build from `/app/client/dist`

### `client/vite.config.ts` *(dev/split mode only)*
- Proxy: `/api/*` → `http://127.0.0.1:8080` (default) or `VITE_QUEUE_API_URL` env var
- WebSocket proxying enabled (`ws: true`)
- Only used when running `npm run dev` or in split-container mode

### `scripts/start.sh`
- Starts Python backend on loopback, waits for `/health`, then starts Nginx
- Detects backend crash during startup
- `wait -n` catches first process death and kills the other
- `trap` handles SIGTERM for clean Docker stop

### `docker-compose.yml`
- `start_period: 120s` — model loading grace period before health checks count
- Health check hits `https://localhost:5173 --insecure` (proves both processes are up)
- GPU reservation via NVIDIA Container Toolkit

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace token for model download |
| `HF_HOME` | No | `/app/.cache/huggingface` | Model cache directory |
| `NO_TORCH_COMPILE` | No | `1` | Set to `1` to skip torch.compile (faster startup) |
| `VITE_QUEUE_API_URL` | No | `http://127.0.0.1:8080` | Backend URL override (for two-container mode) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `exec /app/start.sh: no such file or directory` | Windows line endings in start.sh | Rebuild with `--no-cache` (sed fix is in Dockerfile) |
| Container exits during model load | OOM — not enough VRAM | Need GPU with >= 16GB VRAM |
| `401` on startup | Bad or missing HF_TOKEN | Check `.env` file, accept model license on HuggingFace |
| WebSocket connection fails | Nginx proxy not forwarding WS | Check `nginx.conf` for WebSocket upgrade headers |
| Voice file not found | Filename mismatch | Check `/api/voices` for available files |
| Slow first connection with custom voice | `.wav` being encoded for the first time | Normal — `.pt` cache will be created for next time |
| Health check failing | Model still loading | Wait for `start_period` (120s), check logs |

---

## MCP Integration Pattern

For programmatic control (e.g., from a robot or agent framework), the container exposes a simple WebSocket API. An MCP tool would:

1. Call `GET /api/voices` to list available voices
2. Open `WSS://<host>:5173/api/chat?voice_prompt=<voice>&text_prompt=<instructions>`
3. Pipe audio I/O to hardware (microphone/speaker)
4. Tap `\x02` messages for live transcription
5. Close WebSocket when session ends
6. Reopen with updated `text_prompt` for the next interaction

The model stays warm between connections. Voice `.pt` caches persist across sessions. Only the system prompt text changes.
