# PersonaPlex Unified Technical Specification (v1.1)

This document is the definitive technical reference for building PersonaPlex clients (Mac, Robot, Mobile). All previous documentation is superseded by this specification.

## 1. Discovery & Control Layer (MCP)
The server implements a **Model Context Protocol (MCP)** host over JSON-RPC (HTTP POST).

- **Endpoint**: `POST https://[IP]:8080/mcp`
- **Content-Type**: `application/json`

### Supported Methods
#### `tools/call` -> `list_personas`
Returns all available `.pt` and `.wav` voice profiles.

#### `tools/call` -> `get_model_status`
Returns GPU device info, current VRAM usage, and engine status.

#### `tools/call` -> `start_voice_chat`
**Purpose**: Primes the model for a specific identity.
- **Arguments**: `{"voice_pt_file": "Laura.pt"}`
- **Response**: Returns a JSON object containing the `ws_url` for the binary stream.

---

## 2. Infrastructure & Connection
- **Protocol**: Secure WebSocket (`wss://`)
- **Port**: 8080 (Primary) or 5173 (Proxy)
- **SSL**: Self-signed. Clients **must** disable certificate verification for local development.

---

## 3. The Handshake State Machine
Clients **MUST** follow this sequence. Failure to do so results in **Buffer Backlog**, which manifests as the AI "hearing" audio from the past.

1.  **INIT**: Client opens WebSocket with `voice_prompt` and `text_prompt` query params.
2.  **PRIMING**: The server begins loading the identity.
    - **IF .wav**: Server encodes audio and computes KV-Cache. (15s - 25s)
    - **IF .pt**: Server copies tensors to VRAM. ( < 500ms)
    - **WARNING**: Client must **NOT** send audio during this state.
3.  **READY**: Server sends a single binary byte: **`0x00`**.
4.  **CONVERSING**: Audio streaming is now active.

---

## 4. Audio Specification (The "Golden" Format)
The server uses `sphn.OpusStreamReader` for decoding.

| Requirement | Value | Notes |
|---|---|---|
| **Codec** | **Opus** | Must be encapsulated in **Ogg** (not raw frames). |
| **Sample Rate** | **24,000 Hz** | Strictly required by the Mimi encoder. |
| **Channels** | **1 (Mono)** | Model does not support stereo. |
| **Frame Size** | **80ms** | The AI processes audio in 80ms steps. Client transmission can be 20ms chunks. |
| **Streaming** | **20ms Chunks** | Client SHOULD send 20ms chunks to reduce jitter; Server buffers to 80ms. |
| **Bitrate** | **Variable** | Optimized for VOIP/Speech. |

### Sending Audio (Client → AI)
- Prefix with KIND byte **`0x01`**.
- **Format**: `[0x01] + [Ogg-encapsulated Opus bytes] (24kHz Mono)`.
- **Note**: The AI requires 24kHz. If the client hardware is different (e.g., 16kHz), it matches this rate via resampling before transmission.

### Receiving Audio (AI → Client)
- AI response is prefixed with KIND byte **`0x01`**.
- Format: `[0x01] + [Opus bytes]`.

---

## 5. Metadata Specification
Text transcripts are sent as real-time tokens.

- **Kind**: **`0x02`**
- **Format**: `[0x02] + [UTF-8 String]`
- **Usage**: Displayed as "Laura: [text]" in the UI.

---

## 6. Known Failure Modes
| Symptom | Cause | Correction |
|---|---|---|
| **"Ages of delay"** | Audio sent before `0x00`. | Reset client; wait for handshake before opening mic. |
| **Decoder Error** | Raw Opus sent without Ogg headers. | Use an Ogg muxer (PyAV, etc.) on the client. |
| **404 on /mcp** | Server not restarted. | Ensure the new `server.py` with MCP support is running. |
| **Identity Drift** | Mixing `.wav` and `.pt`. | Stick to `.pt` for repeatable latency. |
