# PersonaPlex Server Container - Quick Setup Script (Windows)

Write-Host "🚀 PersonaPlex Server Container Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  IMPORTANT: Edit .env and add your HF_TOKEN" -ForegroundColor Red
    Write-Host "   Get your token at: https://huggingface.co/settings/tokens"
    Write-Host "   Accept license at: https://huggingface.co/nvidia/personaplex-7b-v1"
    Write-Host ""
    Read-Host "Press Enter after you've added your HF_TOKEN to .env"
}
else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

# Check if HF_TOKEN is set
$envContent = Get-Content .env -Raw
if ($envContent -match "your_huggingface_token_here") {
    Write-Host "❌ ERROR: HF_TOKEN not set in .env" -ForegroundColor Red
    Write-Host "   Please edit .env and add your Hugging Face token"
    exit 1
}

Write-Host "✅ HF_TOKEN configured" -ForegroundColor Green
Write-Host ""

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ ERROR: Docker not found" -ForegroundColor Red
    Write-Host "   Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
}

Write-Host "✅ Docker found" -ForegroundColor Green

# Check NVIDIA GPU
try {
    docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
    Write-Host "✅ NVIDIA GPU accessible" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  WARNING: NVIDIA GPU not accessible" -ForegroundColor Yellow
    Write-Host "   Install NVIDIA Container Toolkit or use Docker Desktop with WSL2"
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

Write-Host ""
Write-Host "🐳 Building and starting containers..." -ForegroundColor Cyan
Write-Host "   This may take 10-15 minutes on first run"
Write-Host ""

docker-compose up --build

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Access PersonaPlex at:"
Write-Host "  Frontend: http://localhost:5173"
Write-Host "  Backend:  http://localhost:8080"
