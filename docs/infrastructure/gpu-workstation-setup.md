# GPU Workstation Setup: Complete Configuration Guide

**Created**: 2025-07-31  
**Environment**: RTX 4090 GPU Workstation via ZeroTier  
**Status**: Production Ready  

---

## 🎯 Overview

This document provides comprehensive setup and configuration for the RTX 4090 GPU workstation used in SAGE (Self-Adaptive Generative Evaluation) development, including network connectivity, authentication, and development environment setup.

## 📋 Hardware Specifications

### GPU Workstation Details
- **Hostname**: `el02`
- **GPU**: NVIDIA GeForce RTX 4090
- **VRAM**: 24,564 MB (24GB)
- **User Account**: `tca`
- **Operating System**: Linux (Ubuntu-based)
- **Python**: 3.12.3
- **Node.js**: v22.17.0

### Network Configuration
- **ZeroTier Network ID**: `db64858fedbf2ce1` (lonely_berners_lee)
- **ZeroTier IP**: `172.25.253.142`
- **Local Network IP**: `192.168.0.111`
- **Connection Type**: DIRECT peer-to-peer (not relayed)
- **Latency**: 7ms (optimal)

## 🔧 Initial Discovery & Diagnosis

### ZeroTier Network Analysis
```bash
# Check ZeroTier network status
sudo zerotier-cli status
# Output: 200 info a2615d3ce2 1.14.2 ONLINE

# List connected networks
sudo zerotier-cli listnetworks
# Output: db64858fedbf2ce1 lonely_berners_lee e2:8e:de:b0:b3:67 OK PRIVATE feth1262 172.25.96.253/16

# Check peer connections
sudo zerotier-cli peers
# Key finding: 8f53f201b7 1.14.2 LEAF 7 DIRECT 594 595 192.168.0.111/25500
```

**Critical Discovery**: ZeroTier automatically detected same LAN and established DIRECT connection at local network speeds.

### SSH Configuration Issues & Resolution

#### Problem Identified
Initial SSH connection failed due to conflicting configuration options in `~/.ssh/config`:
```
RemoteCommand bash
RequestTTY yes
```
These options caused: `Cannot execute command-line and remote command.`

#### Solution Applied
**File**: `/Users/terryli/.ssh/config`
```bash
# Fixed configuration
Host zerotier-remote
    HostName 172.25.253.142
    User tca
    Port 22
    IdentityFile ~/.ssh/id_ed25519_zerotier_np
    StrictHostKeyChecking no
    UserKnownHostsFile ~/.ssh/known_hosts
    ConnectTimeout 5
    ForwardAgent yes
    TCPKeepAlive yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

**Result**: SSH connection successful, supporting both VS Code Remote SSH and direct command execution.

## 🚀 Software Installation & Configuration

### Claude Code Installation Process

#### Prerequisites Verification
```bash
# Check Node.js (already installed)
ssh zerotier-remote "node --version"
# Output: v22.17.0

# Setup npm for user directory
ssh zerotier-remote "
mkdir -p ~/.npm-global
npm config set prefix ~/.npm-global
echo 'export PATH=\$HOME/.npm-global/bin:\$PATH' >> ~/.bashrc
"
```

#### Claude Code Installation
```bash
# Install with correct package name
ssh zerotier-remote "npm install -g @anthropic-ai/claude-code"
# Output: added 3 packages in 6s

# Verify installation
ssh zerotier-remote "
export PATH=~/.npm-global/bin:\$PATH
claude --version
"
# Output: 1.0.64 (Claude Code)
```

### Syncthing Installation

#### User-Space Installation (No sudo required)
```bash
ssh zerotier-remote "
wget -q https://github.com/syncthing/syncthing/releases/download/v1.30.0/syncthing-linux-amd64-v1.30.0.tar.gz
tar -xzf syncthing-linux-amd64-v1.30.0.tar.gz
mkdir -p ~/bin
mv syncthing-linux-amd64-v1.30.0/syncthing ~/bin/
echo 'export PATH=\$HOME/bin:\$PATH' >> ~/.bashrc
~/bin/syncthing --version
"
# Output: syncthing v1.30.0 "Gold Grasshopper"
```

## 🔐 Authentication & Security

### SSH Key Management
- **Key File**: `~/.ssh/id_ed25519_zerotier_np`
- **Key Type**: Ed25519 (high security)
- **Permissions**: 600 (user read/write only)
- **Authentication**: Key-based (no password required)

### ZeroTier Security Features
- **Encryption**: End-to-end encrypted traffic
- **Network Access**: Controlled by network administrator
- **Direct P2P**: No traffic through third-party servers
- **Local Network Detection**: Automatic same-LAN optimization

## 📡 Network Performance Optimization

### ZeroTier Performance Analysis

#### Connection Status Verification
```bash
# Check if connection is DIRECT (not RELAY)
sudo zerotier-cli peers | grep 8f53f201b7
# Output: 8f53f201b7 1.14.2 LEAF 7 DIRECT 594 595 192.168.0.111/25500
```

**Key Performance Indicators**:
- ✅ **Connection Type**: DIRECT (peer-to-peer)
- ✅ **Latency**: 7ms (excellent for same LAN)
- ✅ **Local IP Detection**: 192.168.0.111 (same network)
- ✅ **No Relay Overhead**: Traffic uses local network directly

#### Connectivity Test Results
```bash
# Ping test through ZeroTier
ping -c 2 172.25.253.142
# Results: 2 packets transmitted, 2 received, 0.0% packet loss
# round-trip min/avg/max/stddev = 6.236/8.593/10.950/2.357 ms
```

## 🛠️ Development Environment Setup

### Workspace Directory Structure
```bash
# Create workspace structure
ssh zerotier-remote "mkdir -p ~/eon/nt"

# Verify directory creation
ssh zerotier-remote "ls -la ~/eon/"
# Output: drwxrwxr-x  3 tca tca 4096 Jul 31 02:31 nt
```

### Environment Variables Configuration
```bash
# Add to remote ~/.bashrc
ssh zerotier-remote "cat >> ~/.bashrc << 'EOF'
# Claude Code PATH
export PATH=~/.npm-global/bin:\$PATH

# Syncthing PATH
export PATH=~/bin:\$PATH

# GPU development environment
export CUDA_VISIBLE_DEVICES=0
EOF"
```

## 🔍 GPU Capabilities Verification

### NVIDIA GPU Status
```bash
# Check GPU hardware
ssh zerotier-remote "nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader"
# Output: NVIDIA GeForce RTX 4090, 550.90.07, 24564 MiB

# Check current utilization
ssh zerotier-remote "nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits"
# Output: NVIDIA GeForce RTX 4090, 7, 24564
```

**GPU Specifications**:
- **Model**: NVIDIA GeForce RTX 4090
- **VRAM**: 24GB total
- **Current Usage**: 7MB (idle)
- **Driver**: 550.90.07 (CUDA-compatible)
- **Availability**: Ready for TiRex inference

## ⚡ Quick Access Aliases

### GPU Workstation Aliases
**File**: `~/.claude/gpu-workstation-aliases.sh`
```bash
#!/bin/bash
# GPU Workstation Connection Aliases

# Quick connection aliases
alias gpu='ssh zerotier-remote'
alias gpu-status='ssh zerotier-remote "hostname && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits"'
alias gpu-check='ping -c 2 172.25.253.142 && echo "GPU workstation reachable"'
alias gpu-info='ssh zerotier-remote "echo \"=== GPU Hardware ===\" && nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader && echo \"=== System Info ===\" && uname -a"'

# Development workflow
alias gpu-claude='ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && claude"'
alias gpu-tmux='ssh zerotier-remote -t "tmux new-session -A -s dev"'

# Synchronization helpers
alias gpu-sync-to='rsync -avz --exclude=".git" --exclude="node_modules" --exclude=".venv" ~/eon/nt/ zerotier-remote:~/eon/nt/'
alias gpu-sync-from='rsync -avz --exclude=".git" --exclude="node_modules" --exclude=".venv" zerotier-remote:~/eon/nt/ ~/eon/nt-gpu-backup/'

# Network diagnostics
alias zt-status='sudo zerotier-cli status && sudo zerotier-cli listnetworks'
alias zt-peers='sudo zerotier-cli peers | grep -E "(LEAF|DIRECT)"'
```

**Installation**:
```bash
# Add to shell configuration
echo "source ~/.claude/gpu-workstation-aliases.sh" >> ~/.zshrc
source ~/.claude/gpu-workstation-aliases.sh
```

## 📊 Performance Benchmarking

### Connection Speed Tests
```bash
# Network latency test
for i in {1..5}; do ping -c 1 172.25.253.142 | grep 'time='; done
# Results: Consistent 6-11ms response times

# SSH connection speed
time ssh zerotier-remote "echo 'Connection test'"
# Results: Sub-second connection establishment
```

### File Transfer Performance
```bash
# Test file transfer speed
dd if=/dev/zero of=/tmp/test-1mb bs=1M count=1
time scp /tmp/test-1mb zerotier-remote:/tmp/
# Results: Local network transfer speeds achieved
```

## 🔧 Troubleshooting Solutions

### SSH Connection Issues
**Problem**: SSH commands failing with "Cannot execute command-line and remote command"  
**Root Cause**: Conflicting `RemoteCommand` and `RequestTTY` in SSH config  
**Solution**: Remove conflicting options, keep essential connectivity settings  

### ZeroTier Performance Issues
**Problem**: Slow network performance  
**Diagnosis**: Check if connection is RELAY instead of DIRECT  
**Solution**: Ensure UDP port 9993 is open for LAN discovery  

### Claude Code PATH Issues
**Problem**: `claude: command not found`  
**Solution**: Export PATH in current session and add to ~/.bashrc  
```bash
export PATH=~/.npm-global/bin:$PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
```

## 📈 Monitoring & Maintenance

### Daily Health Checks
```bash
# Quick system status
ssh zerotier-remote "
echo '=== System Status ==='
uptime
echo '=== GPU Status ==='
nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits
echo '=== Disk Space ==='
df -h ~
echo '=== Network Connectivity ==='
ping -c 1 8.8.8.8 > /dev/null && echo 'Internet: OK' || echo 'Internet: FAIL'
"
```

### Log Monitoring
```bash
# Check Syncthing logs
ssh zerotier-remote "tail -20 ~/syncthing.log"

# System log monitoring
ssh zerotier-remote "sudo journalctl -f -u zerotier-one" # (if accessible)
```

## 🎯 Next Steps Integration

### SAGE Model Preparation
The GPU workstation is now ready for:
1. **TiRex Installation**: PyTorch with CUDA support
2. **Model Inference**: Direct GPU acceleration for 35M parameter xLSTM
3. **Uncertainty Quantification**: Real-time confidence scoring
4. **Integration Testing**: With AlphaForge, catch22, tsfresh ensemble

### Development Workflow
```bash
# Complete development session startup
ssh zerotier-remote
cd ~/eon/nt
export PATH=~/.npm-global/bin:$PATH
claude
# Now ready for SAGE development with full AI assistance and GPU access
```

---

## 📋 Configuration Summary

### ✅ Completed Infrastructure
- **Network**: ZeroTier direct P2P connection (7ms latency)
- **Authentication**: SSH key-based access working
- **Development**: Claude Code v1.0.64 installed and configured
- **Synchronization**: Syncthing ready for bidirectional sync
- **GPU**: RTX 4090 available with 24GB VRAM
- **Monitoring**: Comprehensive aliases and health checks

### 🎯 Performance Characteristics
- **Connection Latency**: 6-11ms (local network speeds)
- **SSH Authentication**: Instant key-based access
- **File Transfer**: Local network bandwidth utilization
- **GPU Utilization**: Direct CUDA access without network overhead
- **Development Experience**: Seamless remote development with AI assistance

**Status**: Production-ready infrastructure for SAGE meta-framework development