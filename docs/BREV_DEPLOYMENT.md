# Brev Deployment Guide

## Overview

This guide walks through deploying PersonaPlex on NVIDIA Brev for instant GPU-accelerated access.

---

## Prerequisites

1. **Brev Account**: Sign up at [brev.dev](https://brev.dev)
2. **Hugging Face Token**: 
   - Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Accept PersonaPlex license at [huggingface.co/nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)

---

## Step-by-Step Deployment

### 1. Create Launchable

Navigate to [brev.dev/launchables](https://brev.dev) and click **"Create Launchable"**

### 2. Configure Files and Runtime

**Step 1: Files and Runtime**
- **Runtime Type**: Select **"Docker Compose"**
- **GitHub Repository**: `https://github.com/TwinPeaksTownie/personaplex_server_container`
- **Docker Compose File**: `docker-compose.yml`

### 3. Configure Environment

**Step 2: Configure the Runtime**

Add environment variables:
```
HF_TOKEN=<your_huggingface_token>
NO_TORCH_COMPILE=1
```

### 4. Networking

**Step 3: Jupyter and Networking**

- **Jupyter**: Disable (not needed)
- **Exposed Ports**:
  - `5173` (Unified gateway — Web UI + API)

### 5. Select Compute

**Step 4: Compute Configuration**

Choose based on your needs:

| Option | GPU | VRAM | Cost/hr* | Use Case |
|--------|-----|------|----------|----------|
| **Minimum** | 1x T4 | 16GB | ~$0.50 | Development, testing |
| **Recommended** | 1x A40 | 48GB | ~$1.50 | Production, demos |
| **Premium** | 1x A100 | 80GB | ~$3.00 | High-load production |

*Approximate pricing - check Brev for current rates

### 6. Review and Launch

**Step 5: Final Review**
- Review all settings
- Click **"Create Launchable"**
- Copy the shareable link

---

## Accessing Your Instance

Once deployed, Brev will provide:

1. **Public URL** for the frontend (e.g., `https://your-instance.brev.dev:5173`)
2. **SSH Access** for debugging
3. **Port Forwarding** for local development

### Testing the Deployment

```bash
# Check health (through Nginx gateway)
curl -k https://your-instance.brev.dev:5173/health

# Access frontend
open https://your-instance.brev.dev:5173
```

---

## Managing Your Instance

### Start/Stop via Brev CLI

```bash
# Install Brev CLI
curl -fsSL https://brev.dev/install.sh | bash

# List instances
brev list

# Stop instance (saves costs)
brev stop <instance-id>

# Start instance
brev start <instance-id>
```

### SSH into Instance

```bash
brev ssh <instance-id>

# Once connected, check containers
docker-compose ps
docker-compose logs -f personaplex
```

---

## Updating Your Deployment

### Option 1: Rebuild from GitHub

```bash
# SSH into instance
brev ssh <instance-id>

# Pull latest changes
cd /path/to/personaplex_server_container
git pull

# Rebuild
docker-compose up --build -d
```

### Option 2: Recreate Launchable

1. Update your GitHub repo
2. Create a new Launchable version
3. Launch the new version

---

## Cost Optimization

### Auto-Stop Idle Instances

Configure in Brev dashboard:
- **Idle Timeout**: 30 minutes
- **Auto-Stop**: Enabled

### Use Spot Instances

For non-critical workloads:
- Enable "Spot Instances" in compute selection
- Save up to 70% on GPU costs
- Risk: Instance may be preempted

---

## Troubleshooting

### Container Build Fails

**Symptom**: Build errors during deployment

**Solution**:
```bash
# SSH into instance
brev ssh <instance-id>

# Check build logs
docker-compose logs personaplex

# Common fix: Clear cache
docker-compose down -v
docker system prune -a
docker-compose up --build
```

### Out of Memory

**Symptom**: `CUDA out of memory` errors

**Solutions**:
1. Upgrade to larger GPU (A40 or A100)
2. Enable CPU offload:
   ```yaml
   # In docker-compose.yml
   command: [..., "--cpu-offload"]
   ```

### Ports Not Accessible

**Symptom**: Can't access frontend/backend

**Solution**:
```bash
# Check exposed ports in Brev dashboard
# Ensure firewall rules allow traffic

# Test locally first
brev port-forward <instance-id> 5173:5173
```

---

## Sharing Your Launchable

### Public Link

Share the Launchable URL with anyone:
```
https://brev.dev/launch/<your-launchable-id>
```

Anyone with this link can:
- Deploy their own instance
- Use their own HF_TOKEN
- Customize compute settings

### Private Launchables

For team-only access:
1. Go to Launchable settings
2. Set visibility to "Private"
3. Invite team members via email

---

## Advanced: Custom Voices on Brev

### Upload Voice Files

```bash
# SSH into instance
brev ssh <instance-id>

# Navigate to voices directory
cd /path/to/personaplex_server_container/voices

# Upload your voice file (from local machine)
# On your local machine:
scp Laura.wav <instance-ssh-url>:/path/to/voices/

# Restart backend
docker-compose restart personaplex
```

---

## Support

- **Brev Support**: [brev.dev/docs](https://brev.dev/docs)
- **PersonaPlex Issues**: [github.com/NVIDIA/personaplex/issues](https://github.com/NVIDIA/personaplex/issues)
- **Container Issues**: [github.com/TwinPeaksTownie/personaplex_server_container/issues](https://github.com/TwinPeaksTownie/personaplex_server_container/issues)
