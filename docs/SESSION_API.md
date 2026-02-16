# Session API Reference

The Session API enables tracked voice collaboration sessions with full bidirectional transcript recording. Sessions are designed for AI agent use — start a conversation, let the user talk, then retrieve the transcript for downstream processing.

## Endpoints

### POST /api/sessions/start

Create a new voice collaboration session.

**Request:**
```json
{
  "voice_prompt": "NATF2.pt",
  "text_prompt": "You are a collaborative planning assistant.",
  "text_temperature": 0.7,
  "audio_temperature": 0.8,
  "text_topk": 25,
  "audio_topk": 250
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `voice_prompt` | string | Yes | — | Voice file name from `/api/voices` |
| `text_prompt` | string | No | `""` | System instructions for the AI persona |
| `text_temperature` | float | No | 0.7 | Text generation temperature |
| `audio_temperature` | float | No | 0.8 | Audio generation temperature |
| `text_topk` | int | No | 25 | Text top-k sampling |
| `audio_topk` | int | No | 250 | Audio top-k sampling |

**Response (200):**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "waiting",
  "voice_prompt": "NATF2.pt",
  "text_prompt": "You are a collaborative planning assistant.",
  "transcript": [],
  "created_at": "2026-02-15T10:30:00+00:00",
  "started_at": null,
  "ended_at": null,
  "websocket_url": "/api/chat?voice_prompt=NATF2.pt&text_prompt=...&session_id=a1b2c3d4-...",
  "duration_seconds": null,
  "full_transcript": "",
  "word_count": 0
}
```

**curl:**
```bash
curl -X POST http://localhost:8080/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"voice_prompt": "NATF2.pt", "text_prompt": "You are a helpful assistant."}'
```

---

### GET /api/sessions/{id}

Get session status, transcript, and computed fields.

**Response (200):**
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "completed",
  "voice_prompt": "NATF2.pt",
  "text_prompt": "You are a collaborative planning assistant.",
  "transcript": [
    {"speaker": "ai", "text": "Hello! What would you like to build today?", "t": 0.0},
    {"speaker": "user", "text": "I want to make a real-time dashboard for IoT sensors", "t": 2.3},
    {"speaker": "ai", "text": "That sounds great! What kind of sensors?", "t": 5.1}
  ],
  "created_at": "2026-02-15T10:30:00+00:00",
  "started_at": "2026-02-15T10:30:02+00:00",
  "ended_at": "2026-02-15T10:35:42+00:00",
  "websocket_url": "/api/chat?...",
  "duration_seconds": 340.0,
  "full_transcript": "AI: Hello! What would you like to build today?\nUser: I want to make a real-time dashboard for IoT sensors\nAI: That sounds great! What kind of sensors?",
  "word_count": 27
}
```

**Response (404):**
```json
{"error": "Session not found"}
```

**curl:**
```bash
curl http://localhost:8080/api/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

### POST /api/sessions/{id}/stop

Signal an active session to end gracefully. The WebSocket connection will close shortly after.

**Response (200):** Same format as GET, with `status: "stopped"`.

**Response (400):**
```json
{"error": "Session is already completed"}
```

**curl:**
```bash
curl -X POST http://localhost:8080/api/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/stop
```

---

### GET /api/sessions

List recent sessions (most recent first, max 50).

**Response (200):**
```json
[
  {"session_id": "...", "status": "completed", "duration_seconds": 340.0, ...},
  {"session_id": "...", "status": "waiting", "duration_seconds": null, ...}
]
```

**curl:**
```bash
curl http://localhost:8080/api/sessions
```

---

## Session Lifecycle

```
POST /api/sessions/start
        │
        ▼
   status: "waiting"
        │
        │  (user connects to websocket_url)
        ▼
   status: "active"
        │
        ├──► (WebSocket disconnects naturally)
        │           ▼
        │      status: "completed"
        │
        └──► POST /api/sessions/{id}/stop
                    ▼
               status: "stopped"
```

## Session Data Model

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string (UUID) | Unique session identifier |
| `status` | string | `waiting`, `active`, `completed`, or `stopped` |
| `voice_prompt` | string | Voice file used |
| `text_prompt` | string | System prompt / persona instructions |
| `transcript` | array | Ordered list of transcript entries |
| `created_at` | string (ISO 8601) | When the session was created |
| `started_at` | string or null | When the WebSocket connected |
| `ended_at` | string or null | When the session ended |
| `websocket_url` | string | WebSocket path for the audio client |
| `duration_seconds` | float or null | Computed: ended_at - started_at |
| `full_transcript` | string | Computed: formatted transcript text |
| `word_count` | int | Computed: total words in transcript |

### Transcript Entry

```json
{
  "speaker": "ai",
  "text": "Hello! What would you like to build today?",
  "t": 0.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `speaker` | string | `"ai"` or `"user"` |
| `text` | string | Transcribed text content |
| `t` | float | Seconds since session start |

## MCP Tools

The same functionality is available via the MCP endpoint (`POST /mcp`):

| MCP Tool | REST Equivalent |
|----------|----------------|
| `start_voice_session` | POST /api/sessions/start |
| `get_session` | GET /api/sessions/{id} |
| `stop_session` | POST /api/sessions/{id}/stop |

## Notes

- Sessions are **in-memory only** — container restart clears all sessions
- The GPU is **single-client locked** — only one active voice session at a time
- AI text comes from the model's token stream (high accuracy, real-time)
- User text comes from Whisper ASR transcription (~3-second chunks, slight delay)
- The server runs on plain HTTP port 8080
