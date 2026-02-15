# Bug Report: PersonaPlex Mac Client

> **Author:** Server team  
> **Date:** 2026-02-11  
> **Severity:** Critical — Client cannot connect to backend  
> **Affects:** Mac Mini M4 web client (localhost:8042)

---

## Executive Summary

The Mac client web UI presents voice options that do not exist on the server, causing every connection attempt to fail with `FileNotFoundError`. Additionally, the client UI is configured with an incorrect port (5173 / Vite proxy) instead of directly targeting the backend API (8080). The existing `MAC_CLIENT_DEBUG.md` was written against a now-defunct Nginx gateway architecture and is **completely outdated**.

---

## 🔴 BUG 1: Voice List Shows Unavailable Voices (Critical)

### Observed Behavior
The voice dropdown shows: `NATURAL_F0`, `NATURAL_F1`, `NATURAL_F2`, `NATURAL_F3`, `NATURAL_M0`, `NATURAL_M1`, `NATURAL_M2`, `NATURAL_M3`

When any `NATURAL_*` voice is selected and "Connect" is pressed, the backend immediately crashes:
```
FileNotFoundError: Requested voice prompt 'NATF0.pt' not found in '/app/voices'
```

### Root Cause
The client is either:
1. **Hardcoding a voice list** instead of fetching from the server, OR
2. **Caching a stale voice list** from a previous server configuration, OR
3. **Transforming voice names** incorrectly (`NATURAL_F0` → `NATF0.pt` is wrong; if the file existed it would be `NATURAL_F0.pt`)

### What The Server Actually Has
The `/api/voices` endpoint returns only the files present in `/app/voices`:
```json
["CLAUDE.pt", "LAURA.pt", "Laura.wav"]
```

### Required Fix
1. **Fetch the voice list from the server** at startup by calling:
   ```
   GET https://<PC_IP>:8080/api/voices
   ```
2. **Display ONLY what the server returns.** Do not hardcode or cache voice names.
3. **Use the exact filename** returned by the API as the `voice_prompt` query parameter. No transformations, no truncations.

### Server-Side Validation (For Reference)
```python
# server.py line 135-140
async def handle_list_voices(self, request):
    if self.voice_prompt_dir is None:
        return web.json_response([])
    voices = [f for f in os.listdir(self.voice_prompt_dir)
              if f.lower().endswith(('.pt', '.wav', '.bin'))]
    return web.json_response(voices)
```

---

## 🔴 BUG 2: Wrong Port / Architecture Assumption (Critical)

### Observed Behavior
The client UI shows `PC Brain Server: 192.168.0.194 : 5173`

### Root Cause
The client is targeting port **5173**, which is the Vite dev server (frontend proxy). This is incorrect for the Mac client.

### Current Architecture
```
┌─────────────────────────────────────────────────────┐
│ PC (192.168.0.194)                                  │
│                                                     │
│  ┌────────────────────┐   ┌──────────────────────┐  │
│  │ Vite Frontend      │   │ Python Backend       │  │
│  │ Port 5173          │──>│ Port 8080 (HTTPS)    │  │
│  │ (browser UI only)  │   │ (WebSocket API)      │  │
│  └────────────────────┘   └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ↑                            ↑
    PC browser only            Mac client connects HERE
```

### Required Fix
1. **Change default port from 5173 to 8080.**
2. **Use `wss://` protocol** (not `ws://`) — the backend uses SSL with a self-signed certificate.
3. **Accept self-signed certificates** in the WebSocket client configuration.

### Correct Connection URL
```
wss://192.168.0.194:8080/api/chat?voice_prompt=LAURA.pt&text_prompt=<system>You are a wise and friendly teacher.</system>
```

---

## 🟡 BUG 3: Voice Name Truncation (Medium)

### Observed Behavior
The error message says `NATF0.pt`, but the UI shows `NATURAL_F0`. Something is truncating the voice name before sending it to the server.

### Required Fix
Send the **exact filename** from the `/api/voices` response as the `voice_prompt` parameter. No transformations:
```
# CORRECT
voice_prompt=LAURA.pt

# WRONG
voice_prompt=LAUR.pt
voice_prompt=laura.pt
voice_prompt=LAURA
```

---

## 🟡 BUG 4: Handshake Handling Unclear (Medium)

### Potential Issue
If the client does not wait for the `0x00` handshake byte before sending audio, the connection will fail silently.

### Protocol Sequence
```
1. Client opens WSS connection with query params
2. Server loads voice prompt (.pt embeddings) — takes 2-10 seconds
3. Server sends 0x00 byte (handshake = "ready")
4. Client begins sending [0x01][opus_bytes] audio frames
5. Server responds with [0x01][opus_bytes] and [0x02][utf8_text]
```

### Verification
Confirm the client:
- Waits for a binary message whose **first byte is 0x00** before sending any audio
- Does NOT time out during the voice prompt loading period (can be up to 10 seconds for large .pt files)

---

## Complete API Reference

### Endpoints

| Endpoint | Method | Protocol | Purpose |
|----------|--------|----------|---------|
| `/api/voices` | GET | HTTPS | Returns JSON array of available voice filenames |
| `/api/chat` | GET (WebSocket Upgrade) | WSS | Full-duplex audio chat |

### `/api/chat` Query Parameters

| Parameter | Required | Example | Notes |
|-----------|----------|---------|-------|
| `voice_prompt` | ✅ Yes | `LAURA.pt` | Must exactly match a filename from `/api/voices` |
| `text_prompt` | ✅ Yes | `<system>You are Laura</system>` | System prompt, wrapped in `<system>` tags. Can be empty string. |

**Advanced Debugging:**
- `seed`: (Optional) Omit this for natural conversation. If provided, it must be an integer. Including it forces the AI to be deterministic (not recommended).

### Binary Message Protocol

| Byte Prefix | Direction | Meaning |
|-------------|-----------|---------|
| `0x00` | Server → Client | Handshake: server is ready, client may begin sending audio |
| `0x01` | Both | Audio data: remaining bytes are Ogg-wrapped Opus frames (24kHz mono) |
| `0x02` | Server → Client | Text token: remaining bytes are UTF-8 encoded text |

### Audio Requirements

| Property | Value |
|----------|-------|
| Codec | Opus wrapped in Ogg container (REQUIRED) |
| Sample Rate | 24,000 Hz |
| Channels | Mono (1) |
| Frame Size | 20ms (480 samples) |

**CRITICAL:** The server uses `sphn.OpusStreamReader` which **requires** an Ogg container. Raw Opus frames will be ignored.

---

## Test Checklist

- [ ] Client uses **FFmpeg or PyAV** to generate a valid Ogg Opus stream
- [ ] Client connects to port **8080** (not 5173)
- [ ] Client uses `wss://` protocol
- [ ] Client accepts self-signed SSL certificates  
- [ ] Client waits for `0x00` handshake before sending audio
- [ ] Client sends audio as `[0x01][ogg_opus_bytes]`
- [ ] Client correctly parses response: check first byte for `0x01` (audio) vs `0x02` (text)
- [ ] Client uses **exact filename** from voices API as `voice_prompt` parameter

---

## PyAV Implementation Guide (Python Client)

If you are using PyAV, do not try to send raw packets from the encoder. You must use a muxer to create the Ogg stream:

1. Use a `BytesIO` object as your output file for `av.open(buffer, mode='w', format='ogg')`.
2. Encoded audio packets must be **muxed** into the container.
3. After every `container.mux(packet)`, read the new bytes from your buffer.
4. Prepend `0x01` to these bytes and send them over the WebSocket.
5. This ensures the server receives the `OpusHead` and standard Ogg paging needed to start decoding.
