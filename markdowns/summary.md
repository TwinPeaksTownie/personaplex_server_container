# PersonaPlex Documentation: The "Burn List" (v1.3) - Critical Failure Analysis

This document identifies where our internal documentation has drifted into "Hallucination Territory." The **PERSONAPLEX_TECHNICAL_SPEC.md** is now the ONLY valid source for protocol logic.

## 🚩 The "Lies" and Technical Debt

### [HANDOFF.md](./HANDOFF.md)
> [!CAUTION]
> **Status: FICTIONAL AUTHOR-TRUTH**
> - **The Big Lie**: Claims to be the "Source of Truth," yet specifies **Raw Opus** packets.
> - **The Reality**: The `sphn.OpusStreamReader` in `server.py` **REJECTS** raw Opus. It requires Ogg encapsulation. Following this doc results in silent failure.
> - **The "Unified" Illusion**: Claims a "Unified Gateway" while the implementation runs two ports (8080/5173) and a Node.js dev server inside Docker. It is fragmented and inefficient.

### [REACHY_INTEGRATION.md](./REACHY_INTEGRATION.md)
> [!WARNING]
> **Status: NON-FUNCTIONAL EXAMPLES**
> - **The Big Lie**: Provides Python snippets using `opuslib` to send raw bytes tokind `0x01`.
> - **The Reality**: This code will trigger `Decoder rejected packet` on the server 100% of the time. It is missing the Ogg muxing layer.

### [CLIENT_REFERENCE.md](./CLIENT_REFERENCE.md)
> [!IMPORTANT]
> **Status: PROTOCOL DRIFT**
> - **The Noise**: Continues to validate Ports 8080/5173. While functional for PC, it creates architectural noise and ignores the **Hardware Hardline (Port 8000)** required for robot integration.
> - **The Handshake**: Underspecifies the **Priming Phase**. Sending audio during priming causes "Ages of delay" due to buffer backlog.

### [README.md](./README.md)
> - **Error**: Directs developers to `HANDOFF.md` for binary protocols. This creates a loop of incorrect implementation.

### [PERSONAPLEX_TECHNICAL_SPEC.md](./PERSONAPLEX_TECHNICAL_SPEC.md)
> - **Status: THE ONLY SEMI-FACTUAL DOC**
> - This is the only file that correctly identifies the **Ogg** requirement. Use this for protocol logic, but ignore its port recommendations.

---

## 🛠 Required Corrections
1. **Unify all docs** to specify **Ogg-encapsulated Opus**.
2. **Deprecate Port 5173 (Vite Dev)** in favor of a true unified gateway on **Port 8080** (PC) or **Port 8000** (Robot/Hardware path).
3. **Replace raw bytes examples** in Integration guides with actual Ogg-muxing logic.
