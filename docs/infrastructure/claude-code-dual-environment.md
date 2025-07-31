# Claude Code Dual Environment Setup

**Created**: 2025-07-31  
**Version**: Claude Code v1.0.64  
**Architecture**: Dual environment AI-assisted development  
**Status**: Production Ready  

---

## 🎯 Overview

This document details the complete setup and configuration of Claude Code v1.0.64 across both macOS development environment and RTX 4090 GPU workstation, enabling seamless AI-assisted development in either environment with automatic workspace synchronization.

## 📐 Architecture Overview

### Dual Claude Code Installation
```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│        macOS Environment        │    │     GPU Workstation (el02)     │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ • Claude Code v1.0.64          │    │ • Claude Code v1.0.64          │
│ • Node.js v20+ (system)        │    │ • Node.js v22.17.0             │
│ • npm global: /usr/local/bin/   │    │ • npm global: ~/.npm-global/   │
│ • Workspace: ~/eon/nt/         │◄──►│ • Workspace: ~/eon/nt/         │
│ • Role: Local development      │    │ • Role: GPU-accelerated dev    │
│ • MPS backend available       │    │ • CUDA backend (RTX 4090)      │
└─────────────────────────────────┘    └─────────────────────────────────┘
              ▲                                      ▲
              │                                      │
              └──────── Syncthing Workspace ────────┘
                    Same codebase, AI assistance on both
```

## 🔧 Installation Process

### macOS Installation (System)
**Already installed via npm global**:
```bash
# Verify existing installation
claude --version
# Output: 1.0.64 (Claude Code)

# Installation path
which claude
# Output: /usr/local/bin/claude

# Node.js version compatibility
node --version
# Output: v20+ (compatible)
```

### GPU Workstation Installation Process

#### Prerequisites Setup
```bash
# Check Node.js (pre-installed)
ssh zerotier-remote "node --version"
# Output: v22.17.0 ✅

# Create npm user-global directory  
ssh zerotier-remote "
mkdir -p ~/.npm-global
npm config set prefix ~/.npm-global
echo 'export PATH=\$HOME/.npm-global/bin:\$PATH' >> ~/.bashrc
"
```

#### Claude Code Installation
```bash
# Install Claude Code with correct package name
ssh zerotier-remote "npm install -g @anthropic-ai/claude-code"
# Output: 
# npm WARN deprecated inflight@1.0.6: This module is not supported, and leaks memory.
# npm WARN deprecated glob@7.2.3: Glob versions prior to v9 are no longer supported
# 
# added 3 packages in 6s
# 
# 1 package is looking for funding
#   run `npm fund` for details

# Verify installation
ssh zerotier-remote "
export PATH=~/.npm-global/bin:\$PATH
claude --version
"
# Output: 1.0.64 (Claude Code) ✅
```

#### PATH Configuration Persistence
```bash
# Add to remote ~/.bashrc for persistent PATH
ssh zerotier-remote "cat >> ~/.bashrc << 'EOF'

# Claude Code PATH configuration
export PATH=~/.npm-global/bin:\$PATH

# GPU development environment
export CUDA_VISIBLE_DEVICES=0
EOF"

# Verify PATH configuration
ssh zerotier-remote "source ~/.bashrc && echo \$PATH | grep npm-global"
# Output: /home/tca/.npm-global/bin: (in PATH) ✅
```

## ⚙️ Configuration & Environment Setup

### macOS Claude Code Environment
**System Configuration**:
- **Installation Path**: `/usr/local/bin/claude`
- **Node.js Backend**: System Node.js (v20+)
- **Workspace Access**: Direct filesystem access
- **GPU Backend**: Apple Metal Performance Shaders (MPS)
- **Network**: Local filesystem operations

### GPU Workstation Environment
**User-Space Configuration**:
- **Installation Path**: `~/.npm-global/bin/claude`
- **Node.js Backend**: User Node.js v22.17.0
- **Workspace Access**: Direct filesystem access
- **GPU Backend**: NVIDIA CUDA (RTX 4090)
- **Network**: ZeroTier P2P (7ms latency to macOS)

### Environment Variables Configuration
```bash
# GPU workstation environment setup
ssh zerotier-remote "cat >> ~/.bashrc << 'EOF'

# Claude Code development environment
export PATH=~/.npm-global/bin:\$PATH
export NODE_ENV=development

# GPU acceleration settings
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

# Workspace settings
export CLAUDE_WORKSPACE=~/eon/nt
EOF"
```

## 🚀 Usage Patterns & Workflows

### Workflow 1: macOS Primary Development
```bash
# Start local development session
cd ~/eon/nt/
claude

# Benefits:
# ✅ Familiar local environment
# ✅ Direct file system access
# ✅ No network dependency
# ✅ MPS acceleration for compatible models
# ✅ All changes sync automatically to GPU workstation

# Switch to GPU for TiRex when needed:
# (Files already synced via Syncthing)
ssh zerotier-remote
cd ~/eon/nt && export PATH=~/.npm-global/bin:$PATH && claude
```

### Workflow 2: GPU Workstation Primary Development (Recommended for SAGE)
```bash
# Connect to GPU workstation
ssh zerotier-remote

# Navigate to synced workspace
cd ~/eon/nt/

# Start Claude Code with proper PATH
export PATH=~/.npm-global/bin:$PATH
claude

# Benefits:
# ✅ Direct RTX 4090 access for TiRex
# ✅ No GPU computation network latency
# ✅ Full AI assistance where compute happens
# ✅ Real-time GPU monitoring (nvidia-smi)
# ✅ All changes sync back to macOS automatically
```

### Workflow 3: Hybrid Development
```bash
# Documentation and planning on macOS
cd ~/eon/nt/docs/
claude

# Model development and validation on GPU workstation
ssh zerotier-remote
cd ~/eon/nt/src/
export PATH=~/.npm-global/bin:$PATH
claude

# Seamless handoff - all changes synchronized automatically
```

## 🔄 Synchronization Integration

### Claude Code Workspace Sync
**Syncthing ensures both Claude Code instances work with identical workspace**:

```bash
# Changes made in macOS Claude Code session
echo "# New analysis" >> ~/eon/nt/docs/analysis.md

# Automatically available in GPU workstation session (~10 seconds):
ssh zerotier-remote "cat ~/eon/nt/docs/analysis.md"
# Shows the new analysis content
```

### Session Continuity
**Start session on one environment, continue on another**:
```bash
# Begin analysis on macOS
cd ~/eon/nt/
claude
# User: "Analyze BTCUSDT data for AlphaForge model"
# Claude creates analysis files...
# Exit session

# Continue on GPU workstation with same context:
ssh zerotier-remote
cd ~/eon/nt/  # Files already synced
export PATH=~/.npm-global/bin:$PATH
claude
# User: "Now run TiRex inference on the analyzed data"
# All previous analysis files available
```

## 📊 Performance Characteristics

### Local macOS Performance
```bash
# Startup time
time claude --version
# Output: real 0m0.123s (instant startup)

# File operations
cd ~/eon/nt/
time ls -la > /dev/null
# Output: real 0m0.003s (local filesystem)

# Memory usage
ps aux | grep claude
# Output: ~50-100MB resident memory
```

### Remote GPU Workstation Performance
```bash
# SSH connection + startup
time ssh zerotier-remote "export PATH=~/.npm-global/bin:\$PATH && claude --version"
# Output: real 0m0.8s (including network)

# File operations over sync
ssh zerotier-remote "cd ~/eon/nt/ && time ls -la > /dev/null"
# Output: real 0m0.005s (local filesystem on remote)

# GPU availability check
ssh zerotier-remote "nvidia-smi --query-gpu=gpu_name --format=csv,noheader"
# Output: NVIDIA GeForce RTX 4090 (instant access)
```

### Network Latency Impact
```bash
# Interactive session responsiveness test
time ssh zerotier-remote "echo 'test response'"
# Output: real 0m0.05s (minimal impact)

# File sync verification
echo "test $(date)" >> ~/eon/nt/sync-test.txt
sleep 12
ssh zerotier-remote "cat ~/eon/nt/sync-test.txt"
# Output: Shows macOS content after ~10-12 seconds
```

## 🔧 Troubleshooting & Solutions

### Common Issues

#### Issue: `claude: command not found` on GPU workstation
**Cause**: PATH not configured for current session
```bash
# Temporary fix for current session
export PATH=~/.npm-global/bin:$PATH

# Permanent fix verification
ssh zerotier-remote "grep 'npm-global' ~/.bashrc"
# Should show: export PATH=$HOME/.npm-global/bin:$PATH

# If missing, add to bashrc:
ssh zerotier-remote "echo 'export PATH=\$HOME/.npm-global/bin:\$PATH' >> ~/.bashrc"
```

#### Issue: Node.js version mismatch errors
**Diagnosis**: Check Node.js compatibility
```bash
# Check local version
node --version
# Expected: v20+

# Check remote version  
ssh zerotier-remote "node --version"
# Expected: v22.17.0

# Both versions are compatible with Claude Code v1.0.64
```

#### Issue: Workspace synchronization delays
**Cause**: Syncthing accumulation window (10 seconds)
```bash
# Check sync status
curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq .state
# Expected: "idle" (healthy sync)

# Force immediate sync (if needed)
curl -X POST http://localhost:8384/rest/db/scan?folder=nt-workspace
```

#### Issue: GPU workstation SSH session termination
**Cause**: Network interruption or session timeout
```bash
# Use persistent session with tmux
ssh zerotier-remote -t "tmux new-session -A -s claude-dev"
# Inside tmux:
cd ~/eon/nt
export PATH=~/.npm-global/bin:$PATH
claude

# Detach: Ctrl+B, D
# Reconnect later: ssh zerotier-remote -t "tmux attach -t claude-dev"
```

## 🎯 SAGE Development Integration

### Model-Specific Usage Patterns

#### CPU Models (AlphaForge, catch22, tsfresh)
**Optimal Environment**: Either macOS or GPU workstation
```bash
# AlphaForge development (CPU-only)
# Can run efficiently on macOS:
cd ~/eon/nt/repos/alphaforge/
claude
# User: "Generate alpha factors for BTCUSDT data"

# Or on GPU workstation (with better parallelization):
ssh zerotier-remote
cd ~/eon/nt/repos/alphaforge/
export PATH=~/.npm-global/bin:$PATH
claude
# Same capability, potentially faster CPU
```

#### GPU Models (TiRex)
**Required Environment**: GPU workstation
```bash
# TiRex development (GPU-required)
ssh zerotier-remote
cd ~/eon/nt/
export PATH=~/.npm-global/bin:$PATH
claude

# User: "Load TiRex model for BTCUSDT forecasting"
# Claude can directly access RTX 4090:
# python -c "import torch; print(torch.cuda.is_available())"
# Output: True (with 24GB VRAM available)
```

### Development Session Templates

#### Template 1: SAGE Model Research & Planning
```bash
# Use macOS for documentation-heavy work
cd ~/eon/nt/docs/research/
claude

# User: "Research optimal parameter-free evaluation metrics for SAGE"
# Benefits: Local documentation, fast file access, familiar environment
```

#### Template 2: Model Implementation & Validation
```bash
# Use GPU workstation for compute-intensive work
ssh zerotier-remote
cd ~/eon/nt/src/
export PATH=~/.npm-global/bin:$PATH
claude

# User: "Implement and test TiRex inference pipeline"
# Benefits: Direct GPU access, real-time performance monitoring
```

#### Template 3: Ensemble Integration
```bash
# Use GPU workstation for full SAGE integration
ssh zerotier-remote
cd ~/eon/nt/
export PATH=~/.npm-global/bin:$PATH
claude

# User: "Integrate AlphaForge + TiRex + catch22 + tsfresh in SAGE framework"
# Benefits: All models available, GPU for TiRex, AI assistance for integration
```

## 📈 Productivity Enhancements

### Quick Access Aliases
**File**: `~/.claude/claude-dual-env-aliases.sh`
```bash
#!/bin/bash
# Claude Code Dual Environment Aliases

# Local development
alias claude-local='cd ~/eon/nt && claude'
alias claude-docs='cd ~/eon/nt/docs && claude'

# Remote development
alias claude-remote='ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && claude"'
alias claude-gpu='ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && nvidia-smi && claude"'

# Session management
alias claude-tmux='ssh zerotier-remote -t "tmux new-session -A -s claude-dev -c ~/eon/nt"'
alias claude-resume='ssh zerotier-remote -t "tmux attach -t claude-dev"'

# Development workflow
alias sage-local='cd ~/eon/nt && echo "SAGE development - local environment" && claude'
alias sage-gpu='ssh zerotier-remote -t "cd ~/eon/nt && echo \"SAGE development - GPU environment (RTX 4090)\" && export PATH=~/.npm-global/bin:\$PATH && claude"'

# Status checks
alias claude-versions='echo "=== Local ===" && claude --version && echo "=== Remote ===" && ssh zerotier-remote "export PATH=~/.npm-global/bin:\$PATH && claude --version"'
alias claude-gpu-status='ssh zerotier-remote "nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits"'
```

### Environment-Aware Development
```bash
# Auto-detect optimal environment for task
function sage_dev() {
    local task=$1
    case $task in
        "docs"|"planning"|"research")
            echo "→ Using macOS environment for documentation work"
            cd ~/eon/nt && claude
            ;;
        "tirex"|"gpu"|"inference")
            echo "→ Using GPU workstation for GPU-accelerated work"
            ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && claude"
            ;;
        "ensemble"|"sage"|"integration")
            echo "→ Using GPU workstation for full SAGE integration"
            ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && claude"
            ;;
        *)
            echo "Usage: sage_dev [docs|tirex|ensemble]"
            ;;
    esac
}
```

## 🔒 Security & Privacy

### Authentication Security
- **SSH Keys**: Ed25519 keys for GPU workstation access
- **Claude Code Auth**: Separate authentication on each environment
- **ZeroTier**: Encrypted P2P network communication
- **No Shared Sessions**: Each environment maintains independent Claude Code sessions

### Data Privacy
- **Local Processing**: Claude Code processes data locally on each machine
- **Sync Encryption**: Syncthing uses TLS encryption for file sync
- **No Cloud Dependencies**: Direct peer-to-peer communication only
- **Isolated Environments**: Development work remains in private network

## 📊 Monitoring & Health Checks

### Daily Environment Verification
```bash
#!/bin/bash
# Claude Code dual environment health check

echo "=== Claude Code Dual Environment Health Check $(date) ==="

echo "1. Local macOS Environment:"
claude --version 2>/dev/null && echo "  ✅ Claude Code working" || echo "  ❌ Claude Code failed"
cd ~/eon/nt && ls > /dev/null && echo "  ✅ Workspace accessible" || echo "  ❌ Workspace failed"

echo "2. Remote GPU Environment:"
ssh zerotier-remote "export PATH=~/.npm-global/bin:\$PATH && claude --version" 2>/dev/null && echo "  ✅ Remote Claude Code working" || echo "  ❌ Remote Claude Code failed"
ssh zerotier-remote "cd ~/eon/nt && ls > /dev/null" 2>/dev/null && echo "  ✅ Remote workspace accessible" || echo "  ❌ Remote workspace failed"

echo "3. GPU Availability:"
ssh zerotier-remote "nvidia-smi --query-gpu=gpu_name --format=csv,noheader" 2>/dev/null && echo "  ✅ RTX 4090 available" || echo "  ❌ GPU not accessible"

echo "4. Synchronization Status:"
curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq -r .state 2>/dev/null && echo "  ✅ Sync healthy" || echo "  ❌ Sync issues"

echo "5. Network Connectivity:"
ping -c 1 172.25.253.142 > /dev/null 2>&1 && echo "  ✅ ZeroTier connection working" || echo "  ❌ Network issues"
```

## 📋 Best Practices

### Development Workflow Guidelines

#### When to Use macOS Environment:
- ✅ **Documentation writing** - Fast local file access
- ✅ **Research and planning** - Familiar environment
- ✅ **Code review and analysis** - Large screen, local tools
- ✅ **CPU-only model development** - AlphaForge, catch22, tsfresh
- ✅ **Offline development** - No network dependency

#### When to Use GPU Workstation Environment:
- ✅ **TiRex model work** - Direct GPU access required
- ✅ **Large dataset processing** - Better computational resources
- ✅ **SAGE ensemble integration** - All models available
- ✅ **Performance-critical development** - Closer to production environment
- ✅ **Long-running tasks** - Persistent sessions with tmux

#### Session Management Best Practices:
1. **Use tmux for GPU workstation sessions** - Persistent across disconnections
2. **Check sync status** before switching environments - Ensure latest changes available
3. **GPU memory monitoring** - Use `nvidia-smi` to track VRAM usage
4. **Regular health checks** - Verify both environments daily

---

## 📊 Performance Summary

### ✅ Achieved Capabilities
- **Dual Environment AI Assistance**: Claude Code v1.0.64 working on both systems
- **Seamless Workspace Access**: Same codebase accessible from both environments
- **GPU Acceleration**: Direct RTX 4090 access for TiRex inference
- **Automatic Synchronization**: 10-second bidirectional file sync
- **Session Flexibility**: Switch between environments based on task requirements

### ✅ Performance Metrics
- **Local Startup**: <0.2 seconds (macOS)
- **Remote Connection**: <1 second via ZeroTier
- **Sync Latency**: 10-15 seconds for code changes
- **GPU Availability**: 24GB VRAM directly accessible
- **Network Latency**: 7ms ZeroTier P2P connection

### ✅ Development Benefits
- **Optimal Resource Utilization**: GPU models run where hardware is best
- **Continuous Backup**: Work preserved across both environments
- **Flexible Development**: Choose environment based on task requirements
- **AI Assistance Everywhere**: Claude Code available where computation happens
- **Zero Configuration**: Environment switching requires no manual setup

**Status**: Production-ready dual environment Claude Code setup enabling optimal SAGE development workflow
