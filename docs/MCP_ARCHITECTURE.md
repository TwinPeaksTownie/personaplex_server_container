# PersonaPlex MCP: Streamable HTTP Architecture

> **Note:** This is the **original design document**. The implemented MCP API differs in endpoint and tool names. See [SERVER_UPDATE_REBUILD.md](SERVER_UPDATE_REBUILD.md) for the actual implementation details:
> - Endpoint: `POST /mcp` (not `/mcp/stream`)
> - Implemented tools: `health_check`, `list_voices`, `upload_voice`, `launch_voice_chat`

This document defines the transition of the PersonaPlex Backend from a simple WebSocket demo into a formal **MCP (Model Context Protocol) Server** using **Streamable HTTP**.

## 🎯 The Objective
To expose PersonaPlex's capabilities (Voice Chat, Persona Management, Model Control) as discoverable **Tools** that any MCP-compatible client (Mac, Robot, or LLM) can interact with implicitly.

---

## 🏗️ Protocol Architecture
Instead of traditional polling or rigid REST, we use a **Long-Lived Streamable HTTP Connection** for JSON-RPC communication.

### 1. Transport Layer: Streamable HTTP
*   **Endpoint**: `POST /mcp/stream`
*   **Format**: Line-delimited JSON (NDJSON)
*   **Encoding**: UTF-8
*   **Mechanism**: The server holds the HTTP response open (`Transfer-Encoding: chunked`) and writes JSON-RPC frames directly to the body.

### 2. Message Flow (JSON-RPC 2.0)
The client and server communicate via standard MCP frames over the stream:
```json
{ "jsonrpc": "2.0", "method": "tools/call", "params": { "name": "start_voice_chat", "arguments": { "persona": "Laura" } }, "id": 1 }
```

---

## 🛠️ Tool Manifest (Implicit Capabilities)

The following tools are exposed via the MCP interface:

| Tool Name | Arguments | Description |
| :--- | :--- | :--- |
| `start_voice_chat` | `persona_name`, `voice_pt_file` | Returns the auth token and WebSocket URL for real-time audio. |
| `list_personas` | (none) | Lists all available `.pt` voice embeddings and `.wav` samples. |
| `update_system_prompt` | `prompt_text` | Dynamically updates the role-play instructions without model reload. |
| `upload_voice_sample` | `file_bytes`, `name` | Uploads a new voice reference to the server's VRAM. |
| `get_model_status` | (none) | Returns GPU memory usage, temperature, and "warmup" status. |

---

## 🔒 The "CORS Wall" & Multi-Origin Access
By implementing this at the edge of the container:
1.  **Browser Security**: The CORS headers I added allow JavaScript-based MCP Clients (running in a browser or Electron) to maintain this stream.
2.  **Cross-Platform**: Since it is pure HTTP Stream, a Python script on a Robot or a Swift app on a Mac can consume the tools with standard libraries.

---

## 🚀 Why This Matters
1.  **Discovery**: You don't have to "tell" your robot how to talk to PersonaPlex. The robot connects to the MCP stream, asks for tools, and finds `start_voice_chat`.
2.  **State Persistence**: The HTTP stream maintains the session context. If the stream breaks, the server can safely clean up GPU resources.
3.  **Headless Control**: This completely removes the need for the NVIDIA "Demo UI." The AI model becomes a functional utility in your network.

---

## 📝 Implementation Notes
To reach this state, the backend `server.py` requires:
*   The `mcp` Python library for frame serialization.
*   An asynchronous `StreamResponse` loop for handling the POST body.
*   A Tool Router that maps MCP calls to the existing `Moshi` and `LMGen` logic.
