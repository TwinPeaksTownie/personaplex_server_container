# PersonaPlex Voice Collaboration Skill

## What This Is

A voice conversation tool for AI agents. Instead of crafting the perfect text prompt, the agent launches a natural voice conversation between the user and an AI persona. The full bidirectional transcript (what the user said AND what the AI said) comes back as rich context for planning, building, brainstorming, or any task that benefits from collaborative dialogue.

## When To Use This Skill

- **Before complex builds** — Let the user talk through what they want rather than writing a spec
- **Ambiguous requirements** — A 5-minute voice conversation yields more context than 10 rounds of text clarification
- **Collaborative planning** — Trip planning, event planning, project scoping
- **Brainstorming** — Explore ideas conversationally before committing to a direction
- **User preference gathering** — Tone, style, priorities that are hard to express in text

## Server Requirements

- PersonaPlex server running and accessible (Docker container with GPU)
- Server URL: `https://<host>:5173` (self-signed SSL, use `--insecure` / skip TLS verification)
- GPU with 16GB+ VRAM

## How To Use

### Step 1: Start a Session

```bash
curl -k -X POST https://<host>:5173/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{
    "voice_prompt": "NATF2.pt",
    "text_prompt": "You are a collaborative planning assistant. Help the user think through their idea by asking clarifying questions. Be conversational and encouraging. Summarize key decisions as you go."
  }'
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "waiting",
  "websocket_url": "/api/chat?voice_prompt=NATF2.pt&text_prompt=...&session_id=a1b2c3d4-...",
  ...
}
```

### Step 2: Connect the User

Direct the user's audio client (browser, robot, app) to the WebSocket URL:
```
wss://<host>:5173<websocket_url from step 1>
```

The WebSocket uses a binary protocol:
- Server sends `0x00` = handshake complete, ready for audio
- Client sends `0x01` + Opus audio = user speech
- Server sends `0x01` + Opus audio = AI speech
- Server sends `0x02` + UTF-8 text = live transcription

The user just talks. No typing required.

### Step 3: Poll for Transcript

While the session is active (or after it ends), poll the transcript:

```bash
curl -k https://<host>:5173/api/sessions/<session_id>
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "completed",
  "transcript": [
    {"speaker": "ai", "text": "Hello! What would you like to build today?", "t": 0.0},
    {"speaker": "user", "text": "I want to make a real-time dashboard for IoT sensors", "t": 2.3},
    {"speaker": "ai", "text": "That sounds great! What kind of sensors are we talking about?", "t": 5.1}
  ],
  "full_transcript": "AI: Hello! What would you like to build today?\nUser: I want to make a real-time dashboard...",
  "duration_seconds": 342,
  "word_count": 847
}
```

### Step 4: Stop the Session (Optional)

To end a session programmatically (e.g., after a time limit):

```bash
curl -k -X POST https://<host>:5173/api/sessions/<session_id>/stop
```

Otherwise, the session ends naturally when the WebSocket disconnects.

## MCP Integration

If using the MCP protocol, the same flow works via JSON-RPC:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
  "name": "start_voice_session",
  "arguments": {
    "voice_prompt": "NATF2.pt",
    "text_prompt": "You are a collaborative planning assistant..."
  }
}}
```

Then poll with `get_session`:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
  "name": "get_session",
  "arguments": {"session_id": "a1b2c3d4-..."}
}}
```

## System Prompt Templates

### Build Planning
```
You are a collaborative software architect. Help the user think through what they want to build. Ask about:
- What problem they're solving
- Who the users are
- Key features and priorities
- Technical constraints or preferences
- Timeline and scope
Summarize key decisions as you go. Be encouraging but practical.
```

### Trip/Event Planning
```
You are a friendly travel and event planning assistant. Help the user plan by asking about:
- Dates, duration, and budget
- Preferences and must-haves
- Group size and special needs
- Activities and interests
Keep the conversation natural. Summarize the plan as it forms.
```

### Brainstorming
```
You are a creative brainstorming partner. Help the user explore ideas freely. Build on their suggestions, offer alternatives, and play devil's advocate when useful. Don't judge ideas too quickly — let them flow.
```

### Requirements Gathering
```
You are a business analyst conducting a requirements interview. Ask structured questions to understand:
- Current workflow and pain points
- Desired outcomes and success criteria
- Edge cases and exceptions
- Integration requirements
Document requirements as the user describes them.
```

## Available Voices

Query available voices:
```bash
curl -k https://<host>:5173/api/voices
```

Built-in voices:
- `NATF0.pt` through `NATF3.pt` — Natural female
- `NATM0.pt` through `NATM3.pt` — Natural male
- `VARF0.pt` through `VARF4.pt` — Variety female
- `VARM0.pt` through `VARM4.pt` — Variety male

Custom voices: Upload a `.wav` file via `/api/voices/upload` — it gets encoded on first use and cached as `.pt`.

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voice_prompt` | required | Voice identity file |
| `text_prompt` | `""` | System instructions / persona |
| `text_temperature` | 0.7 | Text creativity (0.1 = focused, 1.0 = creative) |
| `audio_temperature` | 0.8 | Audio expressiveness |
| `text_topk` | 25 | Text vocabulary diversity |
| `audio_topk` | 250 | Audio style diversity |

## Transcript Notes

- **AI speech** is transcribed server-side from the model's text token stream (high accuracy, real-time)
- **User speech** is transcribed via Whisper ASR in ~3-second chunks (good accuracy, slight delay)
- Transcripts are in-memory only — they do not persist across container restarts
- The model is single-client: only one voice session can be active at a time
- Session swap is fast (~0.5-1s) — the model stays warm between sessions
