# PersonaPlex Client Reference (v1.0)

This is the technical specification for building native clients (Mac, Linux, Robot) for the PersonaPlex engine.

## 1. Connection Discovery
- **Primary Endpoint**: `wss://[IP]:8080/api/chat`
- **Proxy Endpoint** (via Vite): `wss://[IP]:5173/api/chat`
- **Voice List**: `https://[IP]:8080/api/voices` (Returns JSON array)

## 2. Query Parameters
| Parameter | Type | Required | Notes |
|---|---|---|---|
| `voice_prompt` | String | **Yes** | Use a `.pt` filename for 500ms startup. `.wav` takes 15s+. |
| `text_prompt` | String | No | System instruction. Wrap in `<system>...</system>`. |
| `seed` | Integer | No | Random seed for inference. |

## 3. Handshake & State Machine
A client **MUST** follow this sequence to avoid the "hallucination delay":

1.  **CONNECT**: Open WSS connection with params.
2.  **PRIMING (Wait)**: The server is loading weights and warming the KV-cache.
    - **CRITICAL**: **Do NOT send audio bytes.** 
    - Anything sent now enters an "Inference Backlog" and causes the AI to respond to your past self.
3.  **READY**: Receive `0x00` (1 byte binary).
4.  **STREAMING**: Send/Receive `0x01` packets.

## 4. Binary Wire Protocol
All messages are binary. Packets are prefixed with a 1-byte **Kind**.

| Kind | Name | Direction | Payload |
|---|---|---|---|
| `0x00` | Handshake | S → C | None. Marks transition to READY. |
| `0x01` | Audio | Both | Ogg-encapsulated Opus Frame. |
| `0x02` | Text | S → C | UTF-8 String (Transcript). |

## 5. Audio Specification
- **Codec**: Opus (Ogg Encapsulated).
- **Sample Rate**: 24,000 Hz (Strict).
- **Channels**: 1 (Mono).
- **Inference Window**: 1920 samples (80ms).
- **Buffer Rule**: Server processes in 80ms chunks. Note: Sending raw Opus frames without Ogg headers will cause the server's decoder to reject packets.

## 6. Logic Errors & Failures
- **Empty Binary**: Server ignores 0-length payloads.
- **Malformed Kind**: Server logs `unknown message kind` and ignores.
- **Decoder Error**: If you send Ogg headers or 44.1kHz audio, the server logs `Decoder rejected packet` and continues, but the AI hears noise.
