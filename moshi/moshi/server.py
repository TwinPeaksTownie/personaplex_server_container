# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
import uuid
from typing import Literal, Optional

import base64
import json
import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Session management for voice collaboration skill
        self.sessions: dict[str, dict] = {}
        self.active_session: Optional[str] = None

        # Whisper ASR for user speech transcription
        self._whisper_model = None
        self._whisper_ready = False
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper 'small' model for user ASR...")
            self._whisper_model = WhisperModel(
                "small",
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if torch.cuda.is_available() else "int8",
            )
            self._whisper_ready = True
            logger.info("Whisper ASR model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Whisper ASR model: {e}. User transcription will be unavailable.")
    
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    # --- Session management endpoints ---

    def _create_session(self, voice_prompt: str, text_prompt: str, **kwargs) -> dict:
        """Create a new session dict and store it."""
        session_id = str(uuid.uuid4())
        text_temp = kwargs.get("text_temperature", 0.7)
        audio_temp = kwargs.get("audio_temperature", 0.8)
        text_topk = kwargs.get("text_topk", 25)
        audio_topk = kwargs.get("audio_topk", 250)

        params = (
            f"voice_prompt={voice_prompt}&text_prompt={text_prompt}"
            f"&text_temperature={text_temp}&audio_temperature={audio_temp}"
            f"&text_topk={text_topk}&audio_topk={audio_topk}"
            f"&pad_mult=0&repetition_penalty=1.0&repetition_penalty_context=64"
            f"&text_seed=-1&audio_seed=-1"
            f"&session_id={session_id}"
        )

        session = {
            "session_id": session_id,
            "status": "waiting",
            "voice_prompt": voice_prompt,
            "text_prompt": text_prompt,
            "transcript": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "ended_at": None,
            "websocket_url": f"/api/chat?{params}",
        }
        self.sessions[session_id] = session
        return session

    def _format_session(self, session: dict) -> dict:
        """Add computed fields to a session for API responses."""
        out = dict(session)
        # Compute duration
        if session["started_at"] and session["ended_at"]:
            try:
                started = datetime.fromisoformat(session["started_at"])
                ended = datetime.fromisoformat(session["ended_at"])
                out["duration_seconds"] = round((ended - started).total_seconds(), 1)
            except Exception:
                out["duration_seconds"] = None
        else:
            out["duration_seconds"] = None
        # Compute full_transcript string
        lines = []
        for entry in session["transcript"]:
            speaker = "AI" if entry["speaker"] == "ai" else "User"
            lines.append(f"{speaker}: {entry['text']}")
        out["full_transcript"] = "\n".join(lines)
        out["word_count"] = sum(len(entry["text"].split()) for entry in session["transcript"])
        return out

    async def handle_session_start(self, request):
        """POST /api/sessions/start — Create a new voice collaboration session."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        voice_prompt = body.get("voice_prompt", "")
        text_prompt = body.get("text_prompt", "")
        if not voice_prompt:
            return web.json_response({"error": "voice_prompt is required"}, status=400)

        # Verify voice file exists
        if self.voice_prompt_dir:
            voice_path = os.path.join(self.voice_prompt_dir, voice_prompt)
            if not os.path.exists(voice_path):
                return web.json_response({"error": f"Voice file '{voice_prompt}' not found"}, status=404)

        session = self._create_session(
            voice_prompt=voice_prompt,
            text_prompt=text_prompt,
            text_temperature=body.get("text_temperature", 0.7),
            audio_temperature=body.get("audio_temperature", 0.8),
            text_topk=body.get("text_topk", 25),
            audio_topk=body.get("audio_topk", 250),
        )
        logger.info(f"Session created: {session['session_id']}")
        return web.json_response(self._format_session(session))

    async def handle_session_get(self, request):
        """GET /api/sessions/{id} — Get session status and transcript."""
        session_id = request.match_info["id"]
        session = self.sessions.get(session_id)
        if not session:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response(self._format_session(session))

    async def handle_session_stop(self, request):
        """POST /api/sessions/{id}/stop — Signal active session to end gracefully."""
        session_id = request.match_info["id"]
        session = self.sessions.get(session_id)
        if not session:
            return web.json_response({"error": "Session not found"}, status=404)
        if session["status"] not in ("waiting", "active"):
            return web.json_response({"error": f"Session is already {session['status']}"}, status=400)
        session["status"] = "stopped"
        session["ended_at"] = datetime.now(timezone.utc).isoformat()
        if self.active_session == session_id:
            self.active_session = None
        logger.info(f"Session stopped: {session_id}")
        return web.json_response(self._format_session(session))

    async def handle_sessions_list(self, request):
        """GET /api/sessions — List recent sessions."""
        sessions = sorted(
            self.sessions.values(),
            key=lambda s: s["created_at"],
            reverse=True,
        )[:50]  # Last 50 sessions
        return web.json_response([self._format_session(s) for s in sessions])

    async def handle_health(self, request):
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def handle_voices(self, request):
        """List available voice prompt files."""
        voices = []
        if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
            for f in sorted(os.listdir(self.voice_prompt_dir)):
                if f.endswith(('.pt', '.wav')):
                    voices.append(f)
        return web.json_response({"voices": voices})

    async def handle_voice_upload(self, request):
        """Upload a .wav file as a new voice prompt."""
        if not self.voice_prompt_dir:
            return web.json_response(
                {"error": "Voice prompt directory not configured"}, status=500
            )

        reader = await request.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"error": "No file uploaded"}, status=400)

        filename = field.filename or "uploaded_voice.wav"
        # Sanitize: only allow alphanumeric, underscore, hyphen, dot
        safe_name = "".join(
            c for c in filename if c.isalnum() or c in ("_", "-", ".")
        )
        if not safe_name.lower().endswith(".wav"):
            return web.json_response(
                {"error": "Only .wav files are accepted"}, status=400
            )

        filepath = os.path.join(self.voice_prompt_dir, safe_name)
        size = 0
        with open(filepath, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > 50 * 1024 * 1024:  # 50MB limit
                    os.remove(filepath)
                    return web.json_response(
                        {"error": "File too large (max 50MB)"}, status=400
                    )
                f.write(chunk)

        logger.info(f"Voice uploaded: {safe_name} ({size} bytes)")
        return web.json_response({"status": "ok", "filename": safe_name, "size": size})

    # --- MCP Streamable HTTP endpoint ---

    MCP_TOOLS = [
        {
            "name": "health_check",
            "description": "Check if the PersonaPlex model is loaded and ready for inference.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_voices",
            "description": "List available voice prompt files (.wav and cached .pt embeddings).",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "upload_voice",
            "description": "Upload a WAV voice prompt file (base64-encoded).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename for the voice (must end in .wav)"},
                    "audio_data": {"type": "string", "description": "Base64-encoded WAV file content"},
                },
                "required": ["filename", "audio_data"],
            },
        },
        {
            "name": "launch_voice_chat",
            "description": (
                "Get the WebSocket URL and parameters for a voice chat session. "
                "Connect to the returned URL to start streaming audio. "
                "The model stays loaded between sessions — only the voice and system prompt change."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "voice_prompt": {"type": "string", "description": "Voice file name (e.g. 'NATF0.pt' or 'custom.wav')"},
                    "text_prompt": {"type": "string", "description": "System instructions for the AI persona", "default": ""},
                    "text_temperature": {"type": "number", "description": "Text generation temperature", "default": 0.7},
                    "audio_temperature": {"type": "number", "description": "Audio generation temperature", "default": 0.8},
                    "text_topk": {"type": "integer", "description": "Text top-k sampling", "default": 25},
                    "audio_topk": {"type": "integer", "description": "Audio top-k sampling", "default": 250},
                },
                "required": ["voice_prompt"],
            },
        },
        {
            "name": "start_voice_session",
            "description": (
                "Start a tracked voice collaboration session with transcript recording. "
                "Returns a session_id and WebSocket URL. After the user connects and talks, "
                "poll get_session to retrieve the full bidirectional transcript (both AI and user speech). "
                "Use this for collaborative planning, brainstorming, or gathering detailed user intent."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "voice_prompt": {"type": "string", "description": "Voice file name (e.g. 'NATF0.pt' or 'custom.wav')"},
                    "text_prompt": {"type": "string", "description": "System instructions for the AI persona", "default": ""},
                    "text_temperature": {"type": "number", "description": "Text generation temperature", "default": 0.7},
                    "audio_temperature": {"type": "number", "description": "Audio generation temperature", "default": 0.8},
                },
                "required": ["voice_prompt"],
            },
        },
        {
            "name": "get_session",
            "description": "Get the status and full transcript of a voice collaboration session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "The session ID returned by start_voice_session"},
                },
                "required": ["session_id"],
            },
        },
        {
            "name": "stop_session",
            "description": "Signal an active voice session to end gracefully.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "The session ID to stop"},
                },
                "required": ["session_id"],
            },
        },
    ]

    async def _mcp_tool_call(self, name: str, arguments: dict) -> dict:
        """Execute an MCP tool and return the result content."""
        if name == "health_check":
            return {"content": [{"type": "text", "text": json.dumps({"status": "ok"})}]}

        elif name == "list_voices":
            voices = []
            if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
                for f in sorted(os.listdir(self.voice_prompt_dir)):
                    if f.endswith(('.pt', '.wav')):
                        voices.append(f)
            return {"content": [{"type": "text", "text": json.dumps({"voices": voices})}]}

        elif name == "upload_voice":
            filename = arguments.get("filename", "")
            audio_data = arguments.get("audio_data", "")
            safe_name = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
            if not safe_name.lower().endswith(".wav"):
                return {"content": [{"type": "text", "text": json.dumps({"error": "Filename must end in .wav"})}], "isError": True}
            if not self.voice_prompt_dir:
                return {"content": [{"type": "text", "text": json.dumps({"error": "Voice prompt directory not configured"})}], "isError": True}
            try:
                raw = base64.b64decode(audio_data)
            except Exception:
                return {"content": [{"type": "text", "text": json.dumps({"error": "Invalid base64 audio data"})}], "isError": True}
            if len(raw) > 50 * 1024 * 1024:
                return {"content": [{"type": "text", "text": json.dumps({"error": "File too large (max 50MB)"})}], "isError": True}
            filepath = os.path.join(self.voice_prompt_dir, safe_name)
            with open(filepath, "wb") as f:
                f.write(raw)
            logger.info(f"Voice uploaded via MCP: {safe_name} ({len(raw)} bytes)")
            return {"content": [{"type": "text", "text": json.dumps({"status": "ok", "filename": safe_name, "size": len(raw)})}]}

        elif name == "launch_voice_chat":
            voice = arguments.get("voice_prompt", "")
            text = arguments.get("text_prompt", "")
            text_temp = arguments.get("text_temperature", 0.7)
            audio_temp = arguments.get("audio_temperature", 0.8)
            text_topk = arguments.get("text_topk", 25)
            audio_topk = arguments.get("audio_topk", 250)
            # Verify voice file exists
            if self.voice_prompt_dir:
                voice_path = os.path.join(self.voice_prompt_dir, voice)
                if not os.path.exists(voice_path):
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Voice file '{voice}' not found"})}], "isError": True}
            params = (
                f"voice_prompt={voice}&text_prompt={text}"
                f"&text_temperature={text_temp}&audio_temperature={audio_temp}"
                f"&text_topk={text_topk}&audio_topk={audio_topk}"
                f"&pad_mult=0&repetition_penalty=1.0&repetition_penalty_context=64"
                f"&text_seed=-1&audio_seed=-1"
            )
            ws_url = f"/api/chat?{params}"
            return {"content": [{"type": "text", "text": json.dumps({
                "websocket_path": ws_url,
                "voice_prompt": voice,
                "text_prompt": text,
                "protocol": {
                    "handshake": "Server sends 0x00 when ready",
                    "client_audio": "0x01 + Opus bytes",
                    "server_audio": "0x01 + Opus bytes",
                    "server_text": "0x02 + UTF-8 transcription",
                },
            })}]}

        elif name == "start_voice_session":
            voice = arguments.get("voice_prompt", "")
            text = arguments.get("text_prompt", "")
            if not voice:
                return {"content": [{"type": "text", "text": json.dumps({"error": "voice_prompt is required"})}], "isError": True}
            if self.voice_prompt_dir:
                voice_path = os.path.join(self.voice_prompt_dir, voice)
                if not os.path.exists(voice_path):
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Voice file '{voice}' not found"})}], "isError": True}
            session = self._create_session(
                voice_prompt=voice,
                text_prompt=text,
                text_temperature=arguments.get("text_temperature", 0.7),
                audio_temperature=arguments.get("audio_temperature", 0.8),
            )
            logger.info(f"Session created via MCP: {session['session_id']}")
            return {"content": [{"type": "text", "text": json.dumps(self._format_session(session))}]}

        elif name == "get_session":
            session_id = arguments.get("session_id", "")
            session = self.sessions.get(session_id)
            if not session:
                return {"content": [{"type": "text", "text": json.dumps({"error": "Session not found"})}], "isError": True}
            return {"content": [{"type": "text", "text": json.dumps(self._format_session(session))}]}

        elif name == "stop_session":
            session_id = arguments.get("session_id", "")
            session = self.sessions.get(session_id)
            if not session:
                return {"content": [{"type": "text", "text": json.dumps({"error": "Session not found"})}], "isError": True}
            if session["status"] not in ("waiting", "active"):
                return {"content": [{"type": "text", "text": json.dumps({"error": f"Session is already {session['status']}"})}], "isError": True}
            session["status"] = "stopped"
            session["ended_at"] = datetime.now(timezone.utc).isoformat()
            if self.active_session == session_id:
                self.active_session = None
            logger.info(f"Session stopped via MCP: {session_id}")
            return {"content": [{"type": "text", "text": json.dumps(self._format_session(session))}]}

        return {"content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {name}"})}], "isError": True}

    async def handle_mcp(self, request):
        """MCP Streamable HTTP endpoint — handles JSON-RPC over POST."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                status=400,
            )

        msg_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "personaplex", "version": "1.0.0"},
            }
        elif method == "notifications/initialized":
            # Client acknowledgement — no response needed but we return success
            return web.json_response({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        elif method == "tools/list":
            result = {"tools": self.MCP_TOOLS}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = await self._mcp_tool_call(tool_name, arguments)
        else:
            return web.json_response(
                {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}},
                status=400,
            )

        return web.json_response({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _whisper_transcribe(self, pcm_float32_24k: np.ndarray) -> str:
        """Run Whisper ASR on a chunk of 24kHz float32 PCM. Returns transcribed text."""
        if not self._whisper_ready or self._whisper_model is None:
            return ""
        try:
            # faster-whisper expects float32 numpy array at 16kHz
            # Resample from 24kHz to 16kHz using simple linear interpolation
            duration = len(pcm_float32_24k) / 24000
            target_len = int(duration * 16000)
            if target_len < 160:  # Less than 10ms of audio, skip
                return ""
            indices = np.linspace(0, len(pcm_float32_24k) - 1, target_len)
            pcm_16k = np.interp(indices, np.arange(len(pcm_float32_24k)), pcm_float32_24k).astype(np.float32)

            segments, _ = self._whisper_model.transcribe(
                pcm_16k,
                beam_size=1,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        except Exception as e:
            logger.warning(f"Whisper transcription error: {e}")
            return ""

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # Session tracking — optional session_id query param
        session_id = request.query.get("session_id")
        session = self.sessions.get(session_id) if session_id else None
        if session:
            session["status"] = "active"
            session["started_at"] = datetime.now(timezone.utc).isoformat()
            self.active_session = session_id
            clog.log("info", f"Session activated: {session_id}")
        session_start_time = time.time()

        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path

        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(request.query["text_prompt"])) if len(request.query["text_prompt"]) > 0 else None
        seed = int(request["seed"]) if "seed" in request.query else None

        # AI text accumulation buffer (tokens arrive one at a time)
        ai_text_buffer = []

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        try:
                            opus_reader.append_bytes(payload)
                        except ValueError:
                            clog.log("error", "Opus decode error: invalid audio packet received (check client codec)")
                            # Do not break; just drop the packet to keep connection alive if possible
                            pass
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            nonlocal ai_text_buffer
            all_pcm_data = None

            # Whisper ASR state: accumulate user PCM for periodic transcription
            user_pcm_buffer = np.array([], dtype=np.float32)
            whisper_interval_samples = int(3.0 * self.mimi.sample_rate)  # ~3 seconds of audio
            last_whisper_time = time.time()

            while True:
                if close:
                    # Flush remaining AI text buffer to transcript
                    if session and ai_text_buffer:
                        text = "".join(ai_text_buffer).strip()
                        if text:
                            session["transcript"].append({
                                "speaker": "ai", "text": text,
                                "t": round(time.time() - session_start_time, 1),
                            })
                        ai_text_buffer.clear()
                    # Flush remaining user audio through Whisper
                    if session and len(user_pcm_buffer) > 4800:  # >0.2s
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, self._whisper_transcribe, user_pcm_buffer
                        )
                        if text:
                            session["transcript"].append({
                                "speaker": "user", "text": text,
                                "t": round(time.time() - session_start_time, 1),
                            })
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm is None or pcm.shape[-1] == 0:
                    continue

                # Accumulate user audio for Whisper ASR
                if session:
                    user_pcm_buffer = np.concatenate((user_pcm_buffer, pcm))

                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                # Periodically run Whisper on accumulated user audio
                if session and self._whisper_ready and len(user_pcm_buffer) >= whisper_interval_samples:
                    now = time.time()
                    if now - last_whisper_time >= 2.0:  # At least 2s between Whisper runs
                        buffer_to_transcribe = user_pcm_buffer.copy()
                        user_pcm_buffer = np.array([], dtype=np.float32)
                        last_whisper_time = now
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, self._whisper_transcribe, buffer_to_transcribe
                        )
                        if text:
                            session["transcript"].append({
                                "speaker": "user", "text": text,
                                "t": round(now - session_start_time, 1),
                            })

                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                            # Accumulate AI text for session transcript
                            if session:
                                ai_text_buffer.append(_text)
                        else:
                            # Token boundary (PAD/EOS) — flush AI text buffer as a transcript entry
                            if session and ai_text_buffer:
                                text = "".join(ai_text_buffer).strip()
                                if text:
                                    session["transcript"].append({
                                        "speaker": "ai", "text": text,
                                        "t": round(time.time() - session_start_time, 1),
                                    })
                                ai_text_buffer.clear()
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if len(request.query["text_prompt"]) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                if close or ws.closed:
                    return False
                # Also check if session was stopped externally
                if session and session["status"] == "stopped":
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")

        # Finalize session on disconnect
        if session and session["status"] == "active":
            session["status"] = "completed"
            session["ended_at"] = datetime.now(timezone.utc).isoformat()
            if self.active_session == session_id:
                self.active_session = None
            clog.log("info", f"Session completed: {session_id} ({len(session['transcript'])} transcript entries)")

        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=True,
    )
    logger.info("warming up the model")
    state.warmup()
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/health", state.handle_health)
    app.router.add_get("/api/voices", state.handle_voices)
    app.router.add_post("/api/voices/upload", state.handle_voice_upload)
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_post("/mcp", state.handle_mcp)
    # Session management endpoints
    app.router.add_post("/api/sessions/start", state.handle_session_start)
    app.router.add_get("/api/sessions/{id}", state.handle_session_get)
    app.router.add_post("/api/sessions/{id}/stop", state.handle_session_stop)
    app.router.add_get("/api/sessions", state.handle_sessions_list)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
