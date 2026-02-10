# PersonaPlex Server Update & Rebuild Guide

## What Changed

The server has been updated from a two-container architecture to a **unified gateway** with a built-in **MCP endpoint**. This document covers the new architecture, what was changed, and how to rebuild.

### Before (Two Containers)
```
Browser → HTTPS :5173 (Vite container)
                 ↓ Docker network
         HTTP :8080 (Python container)
```
Problems: cross-origin WebSocket failures, mixed-content SSL errors, two ports to manage.

### After (Unified Gateway + MCP)
```
Browser/MCP Client → HTTPS :5173 (single container)
                          │
                     Vite proxy
                          │
                    ┌─────┴─────┐
                    │           │
               /api/*       /mcp
               (ws:true)    (POST)
                    │           │
                    └─────┬─────┘
                          │
                   HTTP 127.0.0.1:8080
                   (Python + model in VRAM)
```

One container, one port, SSL terminated at the gateway, MCP-discoverable tools.

---

## Files Changed

| File | Change |
|------|--------|
| `moshi/moshi/server.py` | Added MCP endpoint (`POST /mcp`), enabled voice embedding caching |
| `client/vite.config.ts` | Added `/mcp` proxy, defaulted backend to `localhost:8080`, enabled `ws: true` |
| `Dockerfile.unified` | **NEW** — single container with Python + Node + CUDA |
| `docker-compose.unified.yml` | **NEW** — single-service compose with GPU, volumes, health check |
| `scripts/start.sh` | **NEW** — process orchestrator with health checking |

The original files (`Dockerfile`, `Dockerfile.backend`, `client/Dockerfile`, `docker-compose.yml`, `docker-compose.yaml`) are untouched and still work as a fallback.

---

## Rebuild Commands

### Full rebuild (after code changes)
```powershell
docker compose -f docker-compose.unified.yml build --no-cache
docker compose -f docker-compose.unified.yml up
```

### Quick rebuild (Dockerfile/config only, no code changes)
```powershell
docker compose -f docker-compose.unified.yml build
docker compose -f docker-compose.unified.yml up
```

### Stop and remove
```powershell
docker compose -f docker-compose.unified.yml down
```

### View logs
```powershell
docker compose -f docker-compose.unified.yml logs -f
```

Use `--no-cache` whenever you've changed files that are copied via `COPY` in the Dockerfile (`server.py`, `vite.config.ts`, `start.sh`, etc.). Docker's layer cache may not always detect content changes.

---

## Startup Sequence

When the container starts, `start.sh` runs this sequence:

```
1. Launch Python backend (127.0.0.1:8080)
   ├── Download model weights (first run only, ~14GB)
   ├── Load Mimi audio codec into VRAM
   ├── Load PersonaPlex 7B into VRAM
   ├── Warmup (4 dummy inference steps)
   └── /health returns 200

2. Health check loop (every 2s)
   └── Detects backend crash → exits container

3. Launch Vite gateway (0.0.0.0:5173)
   ├── HTTPS with self-signed cert
   ├── Proxies /api/* → localhost:8080 (WebSocket enabled)
   └── Proxies /mcp → localhost:8080

4. Process supervision
   └── If either process dies → kills the other → exits
```

Expected startup time:
- First run: 2-5 minutes (model download + VRAM loading)
- Subsequent runs: 30-60 seconds (VRAM loading only, weights cached in Docker volume)

---

## MCP Endpoint

The server now exposes an MCP-compatible endpoint at `POST /mcp` using the Streamable HTTP transport. Any MCP client can connect to `https://<host>:5173/mcp`.

### Available Tools

#### `health_check`
Check if the model is loaded and ready.
```json
{"name": "health_check", "arguments": {}}
```
Returns: `{"status": "ok"}`

#### `list_voices`
List available voice files in the voices directory.
```json
{"name": "list_voices", "arguments": {}}
```
Returns: `{"voices": ["NATF0.pt", "custom.wav", ...]}`

#### `upload_voice`
Upload a WAV file as a new voice prompt (base64-encoded).
```json
{"name": "upload_voice", "arguments": {"filename": "my-voice.wav", "audio_data": "<base64>"}}
```
Returns: `{"status": "ok", "filename": "my-voice.wav", "size": 123456}`

#### `launch_voice_chat`
Get the WebSocket URL for a voice chat session with a specific persona.
```json
{
  "name": "launch_voice_chat",
  "arguments": {
    "voice_prompt": "NATF0.pt",
    "text_prompt": "You are a friendly robot assistant.",
    "text_temperature": 0.7,
    "audio_temperature": 0.8
  }
}
```
Returns:
```json
{
  "websocket_path": "/api/chat?voice_prompt=NATF0.pt&text_prompt=You+are+a+friendly+robot+assistant.&...",
  "voice_prompt": "NATF0.pt",
  "text_prompt": "You are a friendly robot assistant.",
  "protocol": {
    "handshake": "Server sends 0x00 when ready",
    "client_audio": "0x01 + Opus bytes",
    "server_audio": "0x01 + Opus bytes",
    "server_text": "0x02 + UTF-8 transcription"
  }
}
```

### Testing MCP with curl

```bash
# Initialize
curl -k -X POST https://localhost:5173/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

# List tools
curl -k -X POST https://localhost:5173/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Call list_voices
curl -k -X POST https://localhost:5173/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_voices","arguments":{}}}'

# Call launch_voice_chat
curl -k -X POST https://localhost:5173/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"launch_voice_chat","arguments":{"voice_prompt":"NATF0.pt","text_prompt":"You are a helpful assistant."}}}'
```

Note: `-k` is required because the gateway uses a self-signed SSL certificate.

---

## Voice Embedding Cache

Voice prompt embedding caching is now enabled (`save_voice_prompt_embeddings=True`). The first time a `.wav` voice file is used:

1. Audio is loaded, normalized to -24 LUFS
2. Encoded through Mimi codec (frame by frame)
3. Each frame runs through the 7B transformer
4. Resulting embeddings + KV cache saved as `<filename>.pt`

Subsequent connections with the same voice skip steps 1-3 and replay the cached embeddings directly. This reduces per-connection voice prompt processing from seconds to milliseconds.

The `.pt` files are saved in the same directory as the `.wav` files (`./voices/`), which is host-mounted, so caches persist across container restarts.

Text system prompts are NOT cached — they're processed per-connection, but this is cheap (just token stepping, no audio encoding).

---

## Ephemeral Persona Swapping Pattern

The model stays loaded in VRAM. Between voice chat sessions:

```
Session 1: voice=laura.pt, prompt="Triage: gather patient info"
    ↓ WebSocket closes
    ↓ Process transcript → distill context
    ↓ (optional: play robot thinking animation during gap)
Session 2: voice=laura.pt, prompt="Patient wants to schedule MRI. Insurance: Blue Cross."
    ↓ WebSocket closes
    ↓ ...repeat
```

| What changes | Reconnect cost |
|-------------|---------------|
| Only system prompt | ~0.5-1s |
| Voice file (cached .pt) | ~0.5-1s |
| Voice file (new .wav, first use) | ~2-10s |

The MCP `launch_voice_chat` tool is called each time with updated `text_prompt` arguments. The voice `.pt` cache is reused automatically.

---

## SSL Considerations

| Layer | Protocol | Certificate |
|-------|----------|-------------|
| External (port 5173) | HTTPS/WSS | Self-signed, generated at build time |
| Internal (loopback 8080) | HTTP/WS | None — plain HTTP over localhost |
| Vite → Python proxy | HTTP | No SSL overhead |

MCP clients connecting to `https://<host>:5173/mcp` must accept the self-signed certificate. Options:
- `curl -k` / `--insecure`
- Configure MCP client to skip TLS verification
- Replace the self-signed cert with a real one (mount via Docker volume to `/app/client/key.pem` and `/app/client/cert.pem`)

---

## Fallback to Two-Container Mode

The original architecture still works:

```powershell
docker compose -f docker-compose.yml up
```

This uses `Dockerfile.backend` + `client/Dockerfile` as two separate containers. The `vite.config.ts` change is backwards-compatible — when `VITE_QUEUE_API_URL` is set (as in the old compose file), it uses that instead of localhost.
