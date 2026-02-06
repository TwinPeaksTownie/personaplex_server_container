#!/bin/bash
# PersonaPlex Server Container - Quick Setup Script

set -e

echo "🚀 PersonaPlex Server Container Setup"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Edit .env and add your HF_TOKEN"
    echo "   Get your token at: https://huggingface.co/settings/tokens"
    echo "   Accept license at: https://huggingface.co/nvidia/personaplex-7b-v1"
    echo ""
    read -p "Press Enter after you've added your HF_TOKEN to .env..."
else
    echo "✅ .env file already exists"
fi

# Check if HF_TOKEN is set
if grep -q "your_huggingface_token_here" .env; then
    echo "❌ ERROR: HF_TOKEN not set in .env"
    echo "   Please edit .env and add your Hugging Face token"
    exit 1
fi

echo "✅ HF_TOKEN configured"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker not found"
    echo "   Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found"

# Check NVIDIA GPU
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: NVIDIA GPU not accessible"
    echo "   Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ NVIDIA GPU accessible"
fi

echo ""
echo "🐳 Building and starting containers..."
echo "   This may take 10-15 minutes on first run"
echo ""

docker-compose up --build

echo ""
echo "✅ Setup complete!"
echo ""
echo "Access PersonaPlex at:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8080"
