# Holistic Gateway Guide (Nginx + PersonaPlex)

This container uses **Nginx** as its primary entrance (the Gateway) on port **5173**. This setup is designed for maximum stability when streaming high-bandwidth audio via WebSockets.

## 🚀 Connection Summary

| Component | Protocol | Endpoint | Description |
|-----------|----------|----------|-------------|
| **Web UI** | HTTPS | `https://localhost:5173` | The React-based control panel. |
| **API/Voice** | WSS | `wss://localhost:5173/api/chat` | The main real-time audio socket. |
| **Direct API**| HTTP | `http://localhost:8080/api/chat` | **(Internal Only)** Bypasses the gateway. |

---

## 🔒 Handling SSL (The Most Critical Step)

Because the gateway uses a **self-signed certificate**, external clients (Mac apps, Python scripts, Robot browsers) will reject the connection by default.

### 1. For Web Browsers
When you first visit `https://localhost:5173`, the browser will show a security warning. Click **"Advanced"** -> **"Proceed to localhost"**. Once you do this, the WebSocket will be able to connect.

### 2. For Mac / Reachy Apps (Swift/Python)
You **must** configure your WebSocket client to skip SSL verification.

**Python (websockets library):**
```python
import ssl
import websockets

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async with websockets.connect("wss://192.168.0.194:5173/api/chat?...", ssl=ssl_context):
    # Success!
```

---

## 📄 Real-time Logs

We have enabled `PYTHONUNBUFFERED=1`. You can now see every detail of the model warmup and inference in real-time.

**To watch the logs:**
```powershell
docker logs -f <container_name>
```

---

## 🛠️ Troubleshooting

### Proxy Timeout
If you see `504 Gateway Timeout`, it means the Python backend is either crashing or taking too long to process a request (rare).
*   **Fix:** Check `docker logs` to see if the Python process died.

### Connection Refused (Port 5173)
Ensure that the Nginx service started successfully inside the container.
*   **Fix:** Run `docker exec -it <container> nginx -t` to verify the config.

### "No matching voice" error
The backend expects voices to be in `/app/voices`. If you mapped a volume, ensure your wav files are actually inside that folder on your host machine.
