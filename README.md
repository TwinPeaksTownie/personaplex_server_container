# PersonaPlex

**An ephemeral voice collaboration tool for AI agents.**

You don't always know what you need before you start talking. Text prompts force you to have a fully formed thought — and whatever you type first locks the conversation into that vector. PersonaPlex gives you a different way in: a short, natural voice conversation that captures what you actually mean, then hands the structured transcript back to an orchestration agent that can do something with it.

It's not a chatbot. It's not a voice assistant you leave running. It's a focused, ephemeral session — 20 seconds to 2 minutes — where a voice persona helps you figure out what you need *right now*, then gets out of the way. A squishy in-between context builder for the moment when you need to talk through something without going through the full TTS/ASR/LLM ceremony, and without poisoning the conversation vector with whatever half-formed thought you'd type into a prompt box.

Built on NVIDIA's [Moshi](https://github.com/kyutai-labs/moshi) full-duplex speech architecture and deployed on [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robotics hardware for the [NVIDIA GTC Golden Ticket Contest](https://developer.nvidia.com/gtc-golden-ticket-contest).

![Architecture](assets/architecture_diagram.png)

## The Idea

Traditional voice pipelines chain ASR → LLM → TTS with latency at every seam. Full-duplex models like Moshi eliminate that — the AI listens and speaks simultaneously at 80ms inference windows, which makes the conversation feel natural enough that you don't have to plan what you're going to say.

PersonaPlex wraps that capability into an MCP-callable tool — a Docker container with a [SKILL.md](SKILL.md) that any orchestration agent can use. It's the next evolution of [Claude-to-Speech](https://github.com/TwinPeaksTownie/Claude-to-Speech), packaged as a containerized skill instead of a local script.

The workflow:

1. **An orchestration agent** creates a session with a voice persona and system prompt
2. **The user talks** — through a browser, robot, or any WebSocket client
3. **The transcript comes back** — both sides, timestamped, as structured data
4. **The agent acts** on what was discussed — building, planning, routing, whatever comes next

Sessions are designed to be short and disposable. The model stays warm between swaps (<500ms to change voice or persona), so an orchestrator can rapidly cycle through different conversation contexts — customer service intake, brainstorming, requirements gathering, topic shifts — without the overhead of a full pipeline restart. Since conversations tend to grow stale as context shifts after about two minutes, the design leans into that constraint: keep sessions ephemeral, swap fast, let the orchestrator manage continuity.

As a product-minded builder, this becomes a manageable, repeatable, adaptable process for commercial applications. If you're simultaneously running an orchestration agent that updates sessions as transcription streams in, PersonaPlex handles the voice while the agent handles the thinking. The container includes Whisper ASR for bidirectional transcription, MCP tool endpoints, and the skill documentation for agent integration.

## Origin Story

This approach started with two Raspberry Pi-powered [Game Boys](https://github.com/TwinPeaksTownie/ClaudeBoy) that needed to talk to each other. The [scene manager](https://github.com/TwinPeaksTownie/scene_manager) built to orchestrate them ran a Director Agent for scene planning and a Turn Manager coordinating dialogue between two Claude instances on separate ports — with N+2 turn buffering, synchronized visual displays, and ElevenLabs TTS that all had to stay coherent simultaneously. It was a significant pain to put together, and most of that pain came from the voice pipeline: ASR, LLM, and TTS as three separate systems that each introduced latency and failure modes.

That experience led to [Claude-to-Speech](https://github.com/TwinPeaksTownie/Claude-to-Speech) as a simpler voice skill, and then to PersonaPlex — where the voice engine is a single full-duplex model that handles listening and speaking in one inference pass. The philosophy: strip it down to the part that actually matters, which is being dynamically adaptable to the context of the moment. Let the orchestration be its own separate concern.

## How It Works

```
┌──────────────┐         ┌──────────────────────────────────┐
│ Orchestration│         │     PersonaPlex Container        │
│    Agent     │────────►│                                  │
│              │  REST   │  Moshi LM ──► Full-duplex voice  │
│  (creates    │  / MCP  │  Mimi Codec ► Real-time audio    │
│   sessions,  │         │  Whisper ───► User transcription  │
│   reads      │◄────────│  Sessions ──► Tracked transcripts │
│   transcripts)  JSON   │                                  │
└──────────────┘         └───────────┬──────────────────────┘
                                     │ WebSocket (binary audio)
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                    ┌────┴────┐            ┌─────┴──────┐
                    │ Browser │            │ Reachy Mini │
                    │   UI    │            │   Robot     │
                    └─────────┘            └────────────┘
```

- **Single container** on port 8080 — Python server, static frontend, all APIs
- **Single GPU** — Moshi, Mimi codec, and Whisper ASR share one device
- **One session at a time** — by design; swap personas in <500ms

## Deployment Options

| Option | Hardware | VRAM | Cost | Notes |
|--------|----------|------|------|-------|
| Local | Your NVIDIA GPU | 16GB+ | Free | RTX 4090, A4000, etc. |
| NVIDIA Brev | A10 | 24GB | ~$0.80/hr | Dedicated instance |
| Shared cloud | A6000 | 48GB | ~$0.60/hr | Shared device, full fidelity |

## Quick Start

```bash
git clone https://github.com/TwinPeaksTownie/personaplex_server_container.git
cd personaplex_server_container

# Add your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env

# Build and run
docker compose up -d --build
```

First start takes ~2 minutes to download model weights and warm up:

```bash
docker logs -f personaplex-personaplex-1
# Wait for: "Running on http://0.0.0.0:8080"
```

### Connect

- **Web UI**: http://localhost:8080
- **WebSocket**: `ws://localhost:8080/api/chat?voice_prompt=Laura.pt&text_prompt=You+are+a+helpful+assistant`
- **MCP**: `POST http://localhost:8080/mcp`
- **Sessions API**: `POST http://localhost:8080/api/sessions/start`

## Session Tracking

The core workflow for agent integration:

```bash
# 1. Create a tracked session
curl -X POST http://localhost:8080/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"voice_prompt": "Laura.pt", "text_prompt": "You are a brainstorming partner."}'
# Returns: {"session_id": "a1b2c3d4-...", "status": "waiting", ...}

# 2. User connects to the WebSocket and talks

# 3. Poll the transcript (while active or after)
curl http://localhost:8080/api/sessions/a1b2c3d4-...
# Returns: {"transcript": [{"speaker": "ai", "text": "...", "t": 0.0}, ...]}

# 4. Session ends when the WebSocket disconnects (or stop it manually)
curl -X POST http://localhost:8080/api/sessions/a1b2c3d4-.../stop
```

AI text comes from the model's token stream (real-time). User speech is transcribed via Whisper ASR in ~3-second chunks. Sessions are in-memory — they don't persist across container restarts, by design.

See [docs/SESSION_API.md](docs/SESSION_API.md) for the full API reference.

## Voices

Three built-in voice personas, stored as pre-computed embeddings for instant loading:

| Voice | File | Notes |
|-------|------|-------|
| Claude | `Claude.pt` | |
| Cole | `Cole.pt` | Voice embedding capturing the essence of Gordon Cole. Expect appearances inside a Reachy Mini around Snoqualmie and North Bend during [The Real Twin Peaks](https://www.facebook.com/TheRealTwinPeaks/) event, Feb 20-22, 2026. |
| Laura | `Laura.pt` | |

Upload custom voices via `/api/voices/upload` — drop a `.wav` and it gets encoded to `.pt` on first use.

## MCP Integration

PersonaPlex exposes an [MCP](https://modelcontextprotocol.io/) endpoint for AI agent tool use:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
  "name": "start_voice_session",
  "arguments": {"voice_prompt": "Laura.pt", "text_prompt": "You are a planning assistant."}
}}
```

Available tools: `list_personas`, `get_model_status`, `start_voice_chat`, `start_voice_session`, `get_session`, `stop_session`

See [SKILL.md](SKILL.md) for the full agent skill documentation.

## Robotics

PersonaPlex powers the voice on Reachy Mini via a companion app:
- **Robot app**: [TwinPeaksTownie/reachy_personaplex](https://huggingface.co/spaces/TwinPeaksTownie/reachy_personaplex) on HuggingFace
- The robot connects over WebSocket, streams mic audio, plays back AI speech
- Track Session toggle in the web UI captures robot conversations as structured transcripts

## Binary Wire Protocol

All WebSocket messages use a 1-byte kind prefix:

| Kind | Name | Direction | Payload |
|------|------|-----------|---------|
| `0x00` | Handshake | Server → Client | Ready signal (wait for this before sending audio) |
| `0x01` | Audio | Both | Ogg-encapsulated Opus (24kHz mono) |
| `0x02` | Text | Server → Client | UTF-8 transcript token |

See [docs/PROTOCOL.md](docs/PROTOCOL.md) and [docs/CLIENT_REFERENCE.md](docs/CLIENT_REFERENCE.md) for full specs.

## Project Structure

```
├── moshi/              # Python backend (Moshi model, server, Whisper ASR)
├── client/             # React/Vite web frontend
├── voices/             # Voice embeddings (.pt files)
├── docs/               # Protocol specs, session API, client reference
├── SKILL.md            # AI agent skill documentation
├── Dockerfile          # Unified container build
├── docker-compose.yml  # GPU-enabled compose config
└── start.sh            # Container entrypoint
```

## Related Projects

- **[Claude-to-Speech](https://github.com/TwinPeaksTownie/Claude-to-Speech)** — Claude Code plugin that embeds invisible TTS markers in responses, auto-detected by a stop hook and sent to ElevenLabs for voice output. PersonaPlex is the full-duplex, containerized evolution — replacing the ASR→LLM→TTS chain with a single speech-to-speech model.
- **[ClaudeBoy](https://github.com/TwinPeaksTownie/ClaudeBoy)** — A Retroflag GPi Case 2 (Game Boy form factor) running a Raspberry Pi CM4 as a portable Claude-powered edge device with voice activation, Pokeball Plus controller, and ElevenLabs TTS. Where the multi-voice orchestration problem first surfaced.
- **[scene_manager](https://github.com/TwinPeaksTownie/scene_manager)** — Multi-agent AI theater system where a Director Agent plans scenes and a Turn Manager coordinates simultaneous dialogue between two Claude instances across separate ports with synchronized visual displays. The orchestration architecture — managing conversation direction, buffering turns, and keeping multiple agents coherent — is the direct ancestor of PersonaPlex's session management design.
- **[reachy_personaplex](https://huggingface.co/spaces/TwinPeaksTownie/reachy_personaplex)** — Companion HuggingFace app for Reachy Mini robotics integration.

## Tech Stack

- **[Moshi](https://github.com/kyutai-labs/moshi)** — Full-duplex speech-to-speech transformer (Kyutai)
- **[Mimi](https://github.com/kyutai-labs/moshi)** — Neural audio codec for real-time encoding/decoding
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2-based Whisper for user speech transcription
- **NVIDIA CUDA 12.4** — GPU inference runtime
- **React + Vite** — Web UI with real-time audio visualization
- **aiohttp** — Async Python server for WebSocket + REST + static serving

## License

MIT — See [LICENSE-MIT](LICENSE-MIT)
