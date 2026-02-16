ARG BASE_IMAGE="nvcr.io/nvidia/cuda"
ARG BASE_IMAGE_TAG="12.4.1-runtime-ubuntu22.04"

# Build stage for frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/client
COPY client/package*.json ./
RUN npm install
COPY client/ ./
RUN npm run build

# Final stage
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Install Node.js for running the dev server (if preferred over static serving)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/moshi/

# Install dependencies first for better caching
COPY moshi/pyproject.toml moshi/uv.lock* /app/moshi/
# Need __init__.py for dynamic version resolution (version = {attr = "moshi.__version__"})
COPY moshi/moshi/__init__.py /app/moshi/moshi/__init__.py
RUN uv venv /app/moshi/.venv --python 3.12
RUN uv sync --no-install-project

# Copy source code later
COPY moshi/ /app/moshi/
RUN uv sync
RUN uv pip install pyloudnorm

# Copy compiled frontend assets only (removes node_modules bloat)
COPY --from=frontend-builder /app/client/dist /app/client/dist

RUN mkdir -p /app/ssl /app/voices

# Copy unified start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Single Entrypoint for both UI and AI
EXPOSE 8080

ENTRYPOINT ["/app/start.sh"]
