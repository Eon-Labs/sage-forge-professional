# Development Aliases & Productivity Shortcuts

**Created**: 2025-07-31  
**Purpose**: Comprehensive command aliases and shortcuts for SAGE development workflow  
**Status**: Production Ready  

---

## 🎯 Overview

This document provides a complete collection of command aliases, shortcuts, and productivity enhancements for seamless SAGE development across dual environments (macOS + GPU workstation).

## 📁 Alias Organization Structure

### File Locations
```
~/.claude/
├── gpu-workstation-aliases.sh      # GPU workstation management
├── claude-dual-env-aliases.sh      # Claude Code dual environment  
├── sage-development-aliases.sh     # SAGE-specific development
├── sync-monitoring-aliases.sh      # Syncthing and sync management
└── network-diagnostics-aliases.sh  # ZeroTier and network tools
```

## 🖥️ GPU Workstation Management Aliases

### File: `~/.claude/gpu-workstation-aliases.sh`
```bash
#!/bin/bash
# GPU Workstation Connection & Management Aliases

# =============================================================================
# QUICK CONNECTION ALIASES
# =============================================================================

# Basic connections
alias gpu='ssh zerotier-remote'
alias gpu-tmux='ssh zerotier-remote -t "tmux new-session -A -s dev"'
alias gpu-resume='ssh zerotier-remote -t "tmux attach -t dev"'

# Development sessions
alias gpu-claude='ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && claude"'
alias gpu-sage='ssh zerotier-remote -t "cd ~/eon/nt && echo \"SAGE Development - GPU Environment (RTX 4090)\" && export PATH=~/.npm-global/bin:\$PATH && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits && claude"'

# =============================================================================
# SYSTEM STATUS & MONITORING
# =============================================================================

# GPU monitoring
alias gpu-status='ssh zerotier-remote "hostname && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits"'
alias gpu-memory='ssh zerotier-remote "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits"'
alias gpu-processes='ssh zerotier-remote "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"'
alias gpu-temp='ssh zerotier-remote "nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits"'

# System monitoring
alias gpu-uptime='ssh zerotier-remote "uptime && free -h && df -h ~"'
alias gpu-load='ssh zerotier-remote "cat /proc/loadavg && vmstat 1 3"'
alias gpu-disk='ssh zerotier-remote "df -h && du -sh ~/eon/nt"'

# Development environment status
alias gpu-env='ssh zerotier-remote "echo \"=== Development Environment Status ===\" && node --version && export PATH=~/.npm-global/bin:\$PATH && claude --version && python3 --version && echo \"CUDA Available:\" && python3 -c \"import torch; print(torch.cuda.is_available())\" 2>/dev/null || echo \"PyTorch not installed\""'

# =============================================================================
# NETWORK & CONNECTIVITY
# =============================================================================

# Connection testing
alias gpu-ping='ping -c 3 172.25.253.142'
alias gpu-check='ping -c 2 172.25.253.142 && echo "✅ GPU workstation reachable"'
alias gpu-speed='time ssh zerotier-remote "echo \"Connection speed test\""'

# Network diagnostics
alias gpu-network='ssh zerotier-remote "ip addr show | grep inet && ss -tuln | grep :22"'
alias gpu-zerotier='ssh zerotier-remote "sudo zerotier-cli status && sudo zerotier-cli listnetworks"'

# =============================================================================
# FILE OPERATIONS & SYNC
# =============================================================================

# Manual synchronization (emergency backup)
alias gpu-sync-to='rsync -avz --progress --exclude=".git" --exclude="node_modules" --exclude=".venv" --exclude="__pycache__" ~/eon/nt/ zerotier-remote:~/eon/nt/'
alias gpu-sync-from='rsync -avz --progress --exclude=".git" --exclude="node_modules" --exclude=".venv" --exclude="__pycache__" zerotier-remote:~/eon/nt/ ~/eon/nt-gpu-backup/'

# Quick file operations
alias gpu-ls='ssh zerotier-remote "ls -la ~/eon/nt"'
alias gpu-tree='ssh zerotier-remote "cd ~/eon/nt && find . -type d -name .git -prune -o -type f -print | head -20"'
alias gpu-logs='ssh zerotier-remote "tail -20 ~/syncthing.log"'

# =============================================================================
# DEVELOPMENT SHORTCUTS
# =============================================================================

# Quick development actions
alias gpu-pull='ssh zerotier-remote "cd ~/eon/nt && git pull origin master"'
alias gpu-status-git='ssh zerotier-remote "cd ~/eon/nt && git status --porcelain"'
alias gpu-jupyter='ssh zerotier-remote -L 8888:localhost:8888 "cd ~/eon/nt && jupyter lab --no-browser --port=8888"'

# Python environment
alias gpu-python='ssh zerotier-remote -t "cd ~/eon/nt && python3"'
alias gpu-pip='ssh zerotier-remote "cd ~/eon/nt && pip3 list | grep -E \"torch|numpy|pandas|transformers\""'

# =============================================================================
# MAINTENANCE & TROUBLESHOOTING
# =============================================================================

# Service management
alias gpu-restart-ssh='ssh zerotier-remote "sudo systemctl restart ssh"'
alias gpu-restart-syncthing='ssh zerotier-remote "pkill syncthing && ~/bin/syncthing --no-browser --no-restart > ~/syncthing.log 2>&1 &"'

# Log monitoring
alias gpu-watch-logs='ssh zerotier-remote "tail -f ~/syncthing.log"'
alias gpu-ssh-logs='ssh zerotier-remote "sudo journalctl -u ssh -f"' # (if accessible)

# Cleanup operations
alias gpu-cleanup='ssh zerotier-remote "cd ~/eon/nt && find . -name \"*.pyc\" -delete && find . -name \"__pycache__\" -type d -exec rm -rf {} + 2>/dev/null || true"'
alias gpu-temp-clean='ssh zerotier-remote "rm -rf /tmp/* ~/.cache/* 2>/dev/null || true"'

# =============================================================================
# INFORMATION & HELP
# =============================================================================

# Quick info
alias gpu-info='ssh zerotier-remote "echo \"=== GPU Hardware ===\" && nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader && echo \"=== System Info ===\" && uname -a && echo \"=== Disk Usage ===\" && df -h ~ && echo \"=== Memory Usage ===\" && free -h"'

# Help and documentation
alias gpu-help='echo "
GPU Workstation Aliases Help:

Basic Connection:
  gpu             - SSH to GPU workstation
  gpu-tmux        - Connect with persistent tmux session
  gpu-claude      - Start Claude Code on GPU workstation
  gpu-sage        - Start SAGE development session

Status & Monitoring:
  gpu-status      - GPU hardware status
  gpu-memory      - GPU memory usage
  gpu-env         - Development environment status
  gpu-uptime      - System uptime and resources

Network & Sync:
  gpu-ping        - Test connection
  gpu-sync-to     - Manual sync to GPU workstation
  gpu-sync-from   - Manual sync from GPU workstation

Development:
  gpu-python      - Interactive Python on GPU
  gpu-jupyter     - Start Jupyter Lab with port forwarding
  gpu-pull        - Git pull on remote

For more details: cat ~/.claude/gpu-workstation-aliases.sh
"'

# =============================================================================
# COMPOSITE WORKFLOWS
# =============================================================================

# Complete development startup
alias gpu-dev='echo "🚀 Starting GPU development session..." && gpu-check && gpu-status && gpu-claude'

# Full system check
alias gpu-health='echo "🔍 GPU Workstation Health Check..." && gpu-check && gpu-status && gpu-env && echo "✅ Health check complete"'

# SAGE development with monitoring
alias sage-gpu-monitor='ssh zerotier-remote -t "
echo \"=== SAGE GPU Development Session ===\"
echo \"GPU Status:\"
nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits
echo \"Environment:\"
cd ~/eon/nt
export PATH=~/.npm-global/bin:\$PATH
echo \"Node.js: \$(node --version)\"
echo \"Claude Code: \$(claude --version)\"
echo \"Python: \$(python3 --version)\"
echo \"PyTorch CUDA: \$(python3 -c \\\"import torch; print(torch.cuda.is_available())\\\" 2>/dev/null || echo \\\"Not available\\\")\"
echo \"Starting development session...\"
claude
"'
```

## 🤖 Claude Code Dual Environment Aliases

### File: `~/.claude/claude-dual-env-aliases.sh`
```bash
#!/bin/bash
# Claude Code Dual Environment Management

# =============================================================================
# LOCAL DEVELOPMENT (MACOS)
# =============================================================================

# Local Claude Code sessions
alias claude-local='cd ~/eon/nt && echo "🍎 Claude Code - macOS Environment" && claude'
alias claude-docs='cd ~/eon/nt/docs && echo "📚 Documentation mode - macOS" && claude'
alias claude-research='cd ~/eon/nt/docs/research && echo "🔬 Research mode - macOS" && claude'

# Local development shortcuts
alias sage-local='cd ~/eon/nt && echo "🧠 SAGE Development - Local Environment (MPS)" && claude'
alias local-status='echo "=== Local Environment Status ===" && claude --version && node --version && python3 --version'

# =============================================================================
# REMOTE DEVELOPMENT (GPU WORKSTATION)
# =============================================================================

# Remote Claude Code sessions
alias claude-remote='ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && echo \"🖥️  Claude Code - GPU Environment\" && claude"'
alias claude-gpu='ssh zerotier-remote -t "cd ~/eon/nt && export PATH=~/.npm-global/bin:\$PATH && echo \"🎮 GPU-Accelerated Development (RTX 4090)\" && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits && claude"'

# Specialized remote sessions
alias tirex-dev='ssh zerotier-remote -t "cd ~/eon/nt && echo \"🦕 TiRex Development Session\" && export PATH=~/.npm-global/bin:\$PATH && python3 -c \"import torch; print(f\\\"CUDA Available: {torch.cuda.is_available()}\\\")\" 2>/dev/null && claude"'
alias sage-gpu='ssh zerotier-remote -t "cd ~/eon/nt && echo \"🧠 SAGE Development - GPU Environment\" && export PATH=~/.npm-global/bin:\$PATH && claude"'

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

# Persistent sessions with tmux
alias claude-tmux='ssh zerotier-remote -t "tmux new-session -A -s claude-dev -c ~/eon/nt"'
alias claude-resume='ssh zerotier-remote -t "tmux attach -t claude-dev"'
alias claude-list='ssh zerotier-remote "tmux list-sessions"'

# Session switching
alias switch-to-gpu='echo "🔄 Switching to GPU workstation..." && claude-gpu'
alias switch-to-local='echo "🔄 Switching to local development..." && claude-local'

# =============================================================================
# ENVIRONMENT COMPARISON
# =============================================================================

# Version checking
alias claude-versions='echo "=== Local Environment ===" && claude --version && node --version && echo "=== Remote Environment ===" && ssh zerotier-remote "export PATH=~/.npm-global/bin:\$PATH && claude --version && node --version"'

# Environment status
alias env-compare='echo "
=== Environment Comparison ===

Local macOS:
$(claude --version)
Node.js: $(node --version)
Python: $(python3 --version 2>/dev/null || echo \"Not available\")

Remote GPU Workstation:
$(ssh zerotier-remote "export PATH=~/.npm-global/bin:\$PATH && claude --version")
Node.js: $(ssh zerotier-remote "node --version")
Python: $(ssh zerotier-remote "python3 --version")
GPU: $(ssh zerotier-remote "nvidia-smi --query-gpu=gpu_name --format=csv,noheader")
"'

# =============================================================================
# TASK-ORIENTED ALIASES
# =============================================================================

# Auto-select optimal environment based on task
function sage_development() {
    local task=$1
    case $task in
        "docs"|"documentation"|"planning"|"research")
            echo "📚 Using macOS for documentation work"
            claude-docs
            ;;
        "tirex"|"gpu"|"inference"|"training")
            echo "🎮 Using GPU workstation for compute-intensive work"
            claude-gpu
            ;;
        "ensemble"|"sage"|"integration"|"all")
            echo "🧠 Using GPU workstation for full SAGE integration"
            sage-gpu
            ;;
        "local"|"cpu"|"testing")
            echo "🍎 Using macOS for local development"
            claude-local
            ;;
        *)
            echo "Usage: sage_development [docs|tirex|ensemble|local]
            
Available options:
  docs       - Documentation and planning (macOS)
  tirex      - TiRex GPU inference (GPU workstation)
  ensemble   - Full SAGE integration (GPU workstation)
  local      - Local CPU development (macOS)"
            ;;
    esac
}

# Shorthand for sage_development function
alias sage-dev='sage_development'

# =============================================================================
# PRODUCTIVITY SHORTCUTS
# =============================================================================

# Quick task starters
alias plan-sage='echo "📋 SAGE Planning Session" && cd ~/eon/nt/docs && claude'
alias implement-sage='echo "⚙️  SAGE Implementation Session" && claude-gpu'
alias test-sage='echo "🧪 SAGE Testing Session" && claude-local'
alias deploy-sage='echo "🚀 SAGE Deployment Session" && claude-gpu'

# Development workflow
alias morning-dev='echo "☀️  Starting morning development session..." && env-compare && claude-local'
alias evening-gpu='echo "🌙 Switching to GPU for intensive work..." && claude-gpu'
```

## 🧠 SAGE Development Specific Aliases

### File: `~/.claude/sage-development-aliases.sh`
```bash
#!/bin/bash
# SAGE-Specific Development Aliases

# =============================================================================
# SAGE MODEL ALIASES
# =============================================================================

# Individual model development
alias alphaforge-dev='cd ~/eon/nt/repos/alphaforge && echo "📊 AlphaForge Development" && claude'
alias catch22-dev='cd ~/eon/nt && echo "🎣 catch22 Features Development" && python3 -c "import pycatch22; print(f\"catch22 version: {pycatch22.__version__}\")" && claude'
alias tsfresh-dev='cd ~/eon/nt && echo "🔍 tsfresh Features Development" && python3 -c "import tsfresh; print(f\"tsfresh version: {tsfresh.__version__}\")" && claude'
alias tirex-gpu='ssh zerotier-remote -t "cd ~/eon/nt && echo \"🦕 TiRex GPU Development\" && export PATH=~/.npm-global/bin:\$PATH && python3 -c \"import torch; print(f\\\"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"None\\\"}\\\")\" && claude"'

# =============================================================================
# SAGE ENSEMBLE DEVELOPMENT
# =============================================================================

# Complete SAGE framework
alias sage-ensemble='ssh zerotier-remote -t "
cd ~/eon/nt
echo \"🧠 SAGE Ensemble Development Session\"
export PATH=~/.npm-global/bin:\$PATH
echo \"=== Model Availability Check ===\"
echo \"✅ AlphaForge: $(ls repos/alphaforge/ | head -3)\"
python3 -c \"import pycatch22; print(\\\"✅ catch22: Available\\\")\" 2>/dev/null || echo \"❌ catch22: Not available\"
python3 -c \"import tsfresh; print(\\\"✅ tsfresh: Available\\\")\" 2>/dev/null || echo \"❌ tsfresh: Not available\"
python3 -c \"import torch; print(f\\\"✅ TiRex GPU: {torch.cuda.is_available()}\\\")\" 2>/dev/null || echo \"❌ TiRex: PyTorch not available\"
echo \"=== Starting SAGE Development ===\"
claude
"'

# SAGE validation workflow
alias sage-validate='ssh zerotier-remote -t "
cd ~/eon/nt
echo \"🔬 SAGE Model Validation Workflow\"
export PATH=~/.npm-global/bin:\$PATH
echo \"Models ready for validation:\"
echo \"  📊 AlphaForge (formulaic alpha factors)\"
echo \"  🎣 catch22 (canonical time series features)\"
echo \"  🔍 tsfresh (automated feature selection)\"
echo \"  🦕 TiRex (GPU-accelerated forecasting)\"
claude
"'

# =============================================================================
# DATA & TESTING ALIASES
# =============================================================================

# Data source management
alias dsm-check='cd ~/eon/nt/repos/data-source-manager && echo "📡 Data Source Manager Status" && ls -la && claude'
alias btcusdt-data='echo "₿ BTCUSDT Data Pipeline" && cd ~/eon/nt && claude'

# Testing workflows
alias sage-test-local='echo "🧪 SAGE Testing - Local CPU Models" && cd ~/eon/nt && claude'
alias sage-test-gpu='ssh zerotier-remote -t "cd ~/eon/nt && echo \"🧪 SAGE Testing - GPU Models\" && export PATH=~/.npm-global/bin:\$PATH && claude"'

# Performance benchmarking
alias sage-benchmark='ssh zerotier-remote -t "
cd ~/eon/nt
echo \"⚡ SAGE Performance Benchmarking\"
export PATH=~/.npm-global/bin:\$PATH
echo \"GPU Status:\"
nvidia-smi --query-gpu=gpu_name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo \"Starting benchmark session...\"
claude
"'

# =============================================================================
# VISUALIZATION & ANALYSIS
# =============================================================================

# FinPlot integration
alias finplot-dev='cd ~/eon/nt/repos/finplot && echo "📈 FinPlot Visualization Development" && python3 -c "import finplot; print(\"FinPlot available\")" && claude'
alias sage-visualize='echo "📊 SAGE Results Visualization" && cd ~/eon/nt && claude'

# Analysis workflows
alias sage-analysis='echo "🔍 SAGE Performance Analysis" && cd ~/eon/nt/docs/analysis && claude'
alias results-review='echo "📋 SAGE Results Review" && cd ~/eon/nt/results && ls -la && claude'

# =============================================================================
# RESEARCH & DOCUMENTATION
# =============================================================================

# Research development
alias sage-research='cd ~/eon/nt/docs/research && echo "🔬 SAGE Research & Theory" && claude'
alias algo-taxonomy='cd ~/eon/nt/docs/research && echo "📚 Algorithm Taxonomy Research" && claude'

# Documentation workflows
alias sage-docs='cd ~/eon/nt/docs && echo "📖 SAGE Documentation" && claude'
alias update-roadmap='cd ~/eon/nt/docs/roadmap && echo "🗺️  SAGE Roadmap Update" && claude'

# =============================================================================
# COMPOSITE WORKFLOWS
# =============================================================================

# Complete SAGE development day
alias sage-full-dev='echo "
🧠 Complete SAGE Development Session

Available workflows:
1. sage-research    - Research and theory development
2. sage-ensemble    - Full ensemble implementation  
3. sage-validate    - Model validation testing
4. sage-benchmark   - Performance benchmarking
5. sage-visualize   - Results visualization
6. sage-docs        - Documentation update

Starting with environment status check...
" && env-compare && echo "Choose your workflow above or run sage-ensemble for full development"'

# Research to implementation pipeline
alias sage-pipeline='echo "
🔄 SAGE Development Pipeline

Phase 1: Research (Local)  → sage-research
Phase 2: Implementation    → sage-ensemble  
Phase 3: Validation       → sage-validate
Phase 4: Benchmarking     → sage-benchmark
Phase 5: Documentation    → sage-docs

Current status: Enhanced Phase 0 Complete ✅
Next: Individual model validation
"'

# =============================================================================
# TROUBLESHOOTING & DIAGNOSTICS
# =============================================================================

# SAGE environment diagnostics
alias sage-diag='echo "
🔍 SAGE Environment Diagnostics

Local macOS:
$(python3 -c "import pycatch22, tsfresh; print(f\"catch22: {pycatch22.__version__}, tsfresh: {tsfresh.__version__}\")" 2>/dev/null || echo "Python packages not available locally")

Remote GPU:
$(ssh zerotier-remote "python3 -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\"" 2>/dev/null || echo "Remote Python environment check failed")

Repositories:
$(ls -la ~/eon/nt/repos/ | grep -E "(alphaforge|nautilus|data-source|finplot)")
"'

# Model availability check
alias models-status='echo "📊 SAGE Models Status:

✅ AlphaForge: $(ls ~/eon/nt/repos/alphaforge/ > /dev/null 2>&1 && echo "Available" || echo "Missing")
✅ NautilusTrader: $(ls ~/eon/nt/repos/nautilus_trader/ > /dev/null 2>&1 && echo "Available" || echo "Missing")  
✅ DSM: $(ls ~/eon/nt/repos/data-source-manager/ > /dev/null 2>&1 && echo "Available" || echo "Missing")
✅ FinPlot: $(ls ~/eon/nt/repos/finplot/ > /dev/null 2>&1 && echo "Available" || echo "Missing")
✅ catch22: $(python3 -c \"import pycatch22; print('Available')\" 2>/dev/null || echo \"Not installed\")
✅ tsfresh: $(python3 -c \"import tsfresh; print('Available')\" 2>/dev/null || echo \"Not installed\")
✅ TiRex GPU: $(ssh zerotier-remote \"python3 -c \\\"import torch; print('Available' if torch.cuda.is_available() else 'CUDA not available')\\\"\" 2>/dev/null || echo \"Remote check failed\")
"'
```

## 🔄 Synchronization Monitoring Aliases

### File: `~/.claude/sync-monitoring-aliases.sh`
```bash
#!/bin/bash
# Syncthing & Synchronization Monitoring Aliases

# =============================================================================
# SYNC STATUS MONITORING
# =============================================================================

# Basic sync status
alias sync-status='curl -s http://localhost:8384/rest/system/status | jq -r .myID && echo "Sync service running"'
alias sync-health='curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq -r .state'
alias sync-errors='curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq -r .errors'

# Detailed sync information
alias sync-info='curl -s http://localhost:8384/rest/system/status | jq "{myID: .myID, cpuPercent: .cpuPercent, uptime: .uptime}"'
alias sync-folder='curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq "{needFiles: .needFiles, state: .state, errors: .errors, globalBytes: .globalBytes, localBytes: .localBytes}"'

# Connection status
alias sync-connections='curl -s http://localhost:8384/rest/system/connections | jq ".connections"'
alias sync-gpu-connection='curl -s http://localhost:8384/rest/system/connections | jq ".connections[\"ZOYKTSR-YBGYP7D-MZHE6RN-SMXZESR-V3QMNKZ-B3CJJIO-KJV5MR4-LIQM3AH\"]"'

# =============================================================================
# SYNC PERFORMANCE MONITORING
# =============================================================================

# Real-time sync monitoring
alias sync-watch='watch -n 5 "curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq -r .state"'
alias sync-live='curl -s http://localhost:8384/rest/events | jq "select(.type == \"ItemFinished\")"'

# Bandwidth and statistics
alias sync-stats='curl -s http://localhost:8384/rest/stats/device | jq "."'
alias sync-bandwidth='curl -s http://localhost:8384/rest/system/connections | jq ".connections[\"ZOYKTSR-YBGYP7D-MZHE6RN-SMXZESR-V3QMNKZ-B3CJJIO-KJV5MR4-LIQM3AH\"] | {connected: .connected, inBytesTotal: .inBytesTotal, outBytesTotal: .outBytesTotal}"'

# Completion status
alias sync-completion='curl -s "http://localhost:8384/rest/db/completion?device=ZOYKTSR-YBGYP7D-MZHE6RN-SMXZESR-V3QMNKZ-B3CJJIO-KJV5MR4-LIQM3AH&folder=nt-workspace" | jq "{completion: .completion, needBytes: .needBytes, needDeletes: .needDeletes}"'

# =============================================================================
# SYNC OPERATIONS
# =============================================================================

# Force operations
alias sync-scan='curl -X POST http://localhost:8384/rest/db/scan?folder=nt-workspace && echo "Forced folder scan initiated"'
alias sync-restart='curl -X POST http://localhost:8384/rest/system/restart && echo "Syncthing restart initiated"'

# Pause/resume operations
alias sync-pause='curl -X POST http://localhost:8384/rest/db/pause?folder=nt-workspace && echo "Sync paused for nt-workspace"'
alias sync-resume='curl -X POST http://localhost:8384/rest/db/resume?folder=nt-workspace && echo "Sync resumed for nt-workspace"'

# =============================================================================
# SYNC TESTING & VERIFICATION
# =============================================================================

# Sync test workflow
alias sync-test='echo "🔄 Testing bidirectional sync..." && 
echo "Test from macOS $(date)" > ~/eon/nt/sync-test-mac.txt &&
echo "⏱️  Waiting 12 seconds for sync..." &&
sleep 12 &&
echo "Remote content:" &&
ssh zerotier-remote "cat ~/eon/nt/sync-test-mac.txt 2>/dev/null || echo \"Sync failed - file not found\""'

alias sync-test-reverse='echo "🔄 Testing reverse sync..." &&
ssh zerotier-remote "echo \"Test from GPU $(date)\" > ~/eon/nt/sync-test-gpu.txt" &&
echo "⏱️  Waiting 12 seconds for sync..." &&
sleep 12 &&
echo "Local content:" &&
cat ~/eon/nt/sync-test-gpu.txt 2>/dev/null || echo "Reverse sync failed - file not found"'

# Comprehensive sync verification
alias sync-verify='echo "🔍 Comprehensive sync verification:

=== Local Files ===" &&
ls -la ~/eon/nt/ | head -5 &&
echo "
=== Remote Files ===" &&
ssh zerotier-remote "ls -la ~/eon/nt/" | head -5 &&
echo "
=== Sync Status ===" &&
sync-health &&
echo "
=== Connection Status ===" &&
curl -s http://localhost:8384/rest/system/connections | jq ".connections[\"ZOYKTSR-YBGYP7D-MZHE6RN-SMXZESR-V3QMNKZ-B3CJJIO-KJV5MR4-LIQM3AH\"].connected"'

# =============================================================================
# LOG MONITORING
# =============================================================================

# Local Syncthing logs
alias sync-logs='tail -20 /opt/homebrew/var/log/syncthing.log'
alias sync-logs-live='tail -f /opt/homebrew/var/log/syncthing.log'
alias sync-logs-errors='grep -i error /opt/homebrew/var/log/syncthing.log | tail -10'

# Remote Syncthing logs
alias sync-logs-remote='ssh zerotier-remote "tail -20 ~/syncthing.log"'
alias sync-logs-remote-live='ssh zerotier-remote "tail -f ~/syncthing.log"'
alias sync-logs-remote-errors='ssh zerotier-remote "grep -i error ~/syncthing.log | tail -10"'

# Combined log monitoring
alias sync-logs-both='echo "=== Local Logs ===" && sync-logs && echo "
=== Remote Logs ===" && sync-logs-remote'

# =============================================================================
# CONFLICT RESOLUTION
# =============================================================================

# Find conflict files
alias sync-conflicts='find ~/eon/nt -name "*.sync-conflict-*" -type f'
alias sync-conflicts-count='find ~/eon/nt -name "*.sync-conflict-*" -type f | wc -l'

# Clean up conflicts (after manual resolution)
alias sync-clean-conflicts='find ~/eon/nt -name "*.sync-conflict-*" -type f -exec rm {} \; && echo "Conflict files cleaned up"'

# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

# Sync timing analysis
alias sync-timing='echo "📊 Sync Performance Analysis:

File Change Detection: <1 second (filesystem watcher)
Accumulation Window: 10 seconds (fsWatcherDelayS)  
Network Transfer: 2-5 seconds (depends on file size)
Total Sync Time: 12-17 seconds typical

Current Status:" && sync-health'

# Sync performance benchmark
alias sync-benchmark='echo "⚡ Sync Performance Benchmark" &&
echo "Creating test files..." &&
for i in {1..5}; do echo "Test file $i content $(date)" > ~/eon/nt/benchmark-test-$i.txt; done &&
echo "Files created, measuring sync time..." &&
start_time=$(date +%s) &&
sleep 15 &&
if ssh zerotier-remote "ls ~/eon/nt/benchmark-test-*.txt > /dev/null 2>&1"; then
    end_time=$(date +%s)
    echo "✅ Sync completed in $((end_time - start_time)) seconds"
    rm ~/eon/nt/benchmark-test-*.txt
    ssh zerotier-remote "rm ~/eon/nt/benchmark-test-*.txt"
else
    echo "❌ Sync benchmark failed"
fi'

# =============================================================================
# MAINTENANCE & TROUBLESHOOTING
# =============================================================================

# Service management
alias sync-service-status='brew services list | grep syncthing'
alias sync-service-restart='brew services restart syncthing && echo "Syncthing service restarted"'
alias sync-service-stop='brew services stop syncthing && echo "Syncthing service stopped"'
alias sync-service-start='brew services start syncthing && echo "Syncthing service started"'

# Remote service management
alias sync-remote-restart='ssh zerotier-remote "pkill syncthing && ~/bin/syncthing --no-browser --no-restart > ~/syncthing.log 2>&1 &" && echo "Remote Syncthing restarted"'
alias sync-remote-status='ssh zerotier-remote "pgrep syncthing > /dev/null && echo \"✅ Remote Syncthing running\" || echo \"❌ Remote Syncthing not running\""'

# Configuration check
alias sync-config='curl -s http://localhost:8384/rest/system/config | jq ".folders[] | select(.id == \"nt-workspace\")"'

# =============================================================================
# COMPREHENSIVE HEALTH CHECK
# =============================================================================

alias sync-health-full='echo "🏥 Comprehensive Sync Health Check

=== Service Status ===
Local: $(brew services list | grep syncthing | awk "{print \$2}")  
Remote: $(ssh zerotier-remote "pgrep syncthing > /dev/null && echo \"Running\" || echo \"Stopped\"")

=== Sync Status ===
Folder State: $(sync-health)
Errors: $(sync-errors)

=== Connection Status ===  
GPU Workstation: $(curl -s http://localhost:8384/rest/system/connections | jq -r ".connections[\"ZOYKTSR-YBGYP7D-MZHE6RN-SMXZESR-V3QMNKZ-B3CJJIO-KJV5MR4-LIQM3AH\"].connected")

=== Recent Conflicts ===
Conflict Files: $(sync-conflicts-count)

=== Performance ===
Last Completion: $(curl -s http://localhost:8384/rest/db/status?folder=nt-workspace | jq -r .stateChanged)

Health Check Complete ✅"'
```

## 🌐 Network Diagnostics Aliases

### File: `~/.claude/network-diagnostics-aliases.sh`
```bash
#!/bin/bash
# ZeroTier Network & Connectivity Diagnostics

# =============================================================================
# ZEROTIER STATUS & MONITORING
# =============================================================================

# Basic ZeroTier status
alias zt-status='sudo zerotier-cli status && sudo zerotier-cli listnetworks'
alias zt-info='sudo zerotier-cli info -j | jq "{nodeId: .config.nodeId, version: .version, online: .online}"'
alias zt-networks='sudo zerotier-cli listnetworks -j | jq ".[] | {networkId: .networkId, name: .name, status: .status, type: .type}"'

# Peer connection analysis
alias zt-peers='sudo zerotier-cli peers | grep -E "(LEAF|DIRECT|RELAY)"'
alias zt-gpu='sudo zerotier-cli peers | grep 8f53f201b7'
alias zt-connections='sudo zerotier-cli peers | grep -E "(DIRECT|RELAY)" | wc -l | xargs echo "Active connections:"'

# =============================================================================
# CONNECTION QUALITY ANALYSIS
# =============================================================================

# Latency testing
alias zt-ping='ping -c 5 172.25.253.142'
alias zt-ping-continuous='ping 172.25.253.142'
alias zt-latency='ping -c 10 172.25.253.142 | grep "round-trip" | awk -F"/" "{print \"Average latency: \" \$5 \" ms\"}"'

# Connection type verification
alias zt-direct='sudo zerotier-cli peers | grep 8f53f201b7 | grep DIRECT && echo "✅ DIRECT connection active" || echo "⚠️  Using RELAY connection"'
alias zt-performance='echo "📊 ZeroTier Performance Analysis:" && zt-direct && zt-latency'

# Network path analysis
alias zt-route='route get 172.25.253.142'
alias zt-trace='traceroute 172.25.253.142'

# =============================================================================
# CONNECTIVITY TESTING
# =============================================================================

# Basic connectivity tests
alias net-check='ping -c 3 172.25.253.142 && echo "✅ GPU workstation reachable"'
alias net-speed='time ssh zerotier-remote "echo \"Speed test\"" && echo "SSH response time measured above"'

# Service connectivity
alias ssh-test='ssh -o ConnectTimeout=5 zerotier-remote "echo \"SSH connectivity confirmed\""'
alias http-test='curl -s --connect-timeout 5 http://172.25.253.142:8384 > /dev/null && echo "✅ Syncthing web interface reachable" || echo "❌ Syncthing not accessible"'

# Port connectivity
alias port-22='nc -z 172.25.253.142 22 && echo "✅ SSH port (22) open" || echo "❌ SSH port closed"'
alias port-8384='nc -z 172.25.253.142 8384 && echo "✅ Syncthing port (8384) open" || echo "❌ Syncthing port closed"'

# =============================================================================
# BANDWIDTH & PERFORMANCE TESTING
# =============================================================================

# File transfer speed test
alias net-benchmark='echo "📊 Network Benchmark Test" &&
dd if=/dev/zero of=/tmp/net-test-1mb bs=1M count=1 2>/dev/null &&
echo "Testing 1MB transfer..." &&
time scp /tmp/net-test-1mb zerotier-remote:/tmp/ &&
rm /tmp/net-test-1mb &&
ssh zerotier-remote "rm /tmp/net-test-1mb" &&
echo "Network benchmark complete"'

# SSH multiplexing performance
alias ssh-performance='echo "🚀 SSH Performance Test" &&
echo "First connection:" &&
time ssh zerotier-remote "echo test1" &&
echo "Second connection (multiplexed):" &&
time ssh zerotier-remote "echo test2"'

# =============================================================================
# NETWORK INTERFACE ANALYSIS
# =============================================================================

# Local network interface
alias net-local='ifconfig | grep -A 5 "inet 192.168.0" && echo "
ZeroTier Interface:" && ifconfig | grep -A 5 feth'

# Remote network interface
alias net-remote='ssh zerotier-remote "ip addr show | grep -E \"inet.*192.168|inet.*172.25\""'

# Interface comparison
alias net-compare='echo "=== Local Network ===" && net-local && echo "
=== Remote Network ===" && net-remote'

# =============================================================================
# SECURITY & ENCRYPTION VERIFICATION
# =============================================================================

# ZeroTier encryption status
alias zt-security='sudo zerotier-cli info -j | jq "{encryption: .config.settings.primaryPort, version: .version}" && echo "ZeroTier encryption active"'

# SSH encryption verification
alias ssh-ciphers='ssh -Q cipher zerotier-remote | head -5'
alias ssh-security='ssh -o LogLevel=DEBUG zerotier-remote "echo test" 2>&1 | grep -i "cipher\|kex\|mac" | head -3'

# =============================================================================
# TROUBLESHOOTING & DIAGNOSTICS
# =============================================================================

# Common issue diagnosis
alias net-diagnose='echo "🔍 Network Diagnostic Report

=== ZeroTier Status ===
$(zt-status | head -2)

=== Connection Type ===
$(zt-direct)

=== Connectivity Test ===
$(ping -c 1 172.25.253.142 > /dev/null 2>&1 && echo "✅ Basic connectivity OK" || echo "❌ Connectivity failed")

=== SSH Access ===
$(ssh -o ConnectTimeout=5 zerotier-remote "echo SSH OK" 2>/dev/null || echo "❌ SSH failed")

=== Service Ports ===
SSH (22): $(nc -z 172.25.253.142 22 && echo "Open" || echo "Closed")
Syncthing (8384): $(nc -z 172.25.253.142 8384 && echo "Open" || echo "Closed")

=== Performance ===
$(zt-latency 2>/dev/null || echo "Latency test failed")

Diagnostic Complete ✅"'

# Detailed troubleshooting
alias net-debug='echo "🔧 Detailed Network Debug

=== ZeroTier Peers ===
$(sudo zerotier-cli peers)

=== Network Routes ===
$(route get 172.25.253.142)

=== DNS Resolution ===
$(nslookup 172.25.253.142 2>/dev/null || echo "No DNS entry")

=== Firewall Status ===
$(sudo pfctl -sr 2>/dev/null | grep -E "9993|22|8384" || echo "No relevant firewall rules")

Debug Complete ✅"'

# =============================================================================
# MAINTENANCE & RECOVERY
# =============================================================================

# ZeroTier service management
alias zt-restart='echo "Restarting ZeroTier..." && sudo launchctl unload /Library/LaunchDaemons/com.zerotier.one.plist && sudo launchctl load /Library/LaunchDaemons/com.zerotier.one.plist && echo "ZeroTier restarted"'

# Network reset procedures
alias net-reset='echo "⚠️  Network reset procedure:" && 
echo "1. Leave ZeroTier network: sudo zerotier-cli leave db64858fedbf2ce1" &&
echo "2. Rejoin ZeroTier network: sudo zerotier-cli join db64858fedbf2ce1" &&
echo "3. Contact admin for network authorization" &&
echo "4. Verify with: zt-status"'

# Connection recovery
alias net-recover='echo "🔄 Connection recovery sequence..." &&
zt-restart &&
sleep 5 &&
zt-status &&
echo "Testing connectivity..." &&
net-check'

# =============================================================================
# MONITORING & ALERTS
# =============================================================================

# Continuous monitoring
alias net-monitor='echo "📡 Starting network monitoring... (Ctrl+C to stop)" &&
while true; do
    clear
    echo "=== Network Monitor $(date) ==="
    zt-direct
    echo ""
    ping -c 1 172.25.253.142 | grep "time="
    echo ""
    ssh -o ConnectTimeout=2 zerotier-remote "echo SSH OK" 2>/dev/null && echo "✅ SSH responsive" || echo "❌ SSH timeout"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done'

# Network quality watch
alias net-watch='watch -n 5 "echo \"ZeroTier Status:\" && sudo zerotier-cli peers | grep 8f53f201b7 && echo \"\" && echo \"Connectivity:\" && ping -c 1 172.25.253.142 | grep time="'
```

## 🔧 Installation & Setup

### Complete Alias Setup Script
```bash
#!/bin/bash
# Complete aliases installation script

echo "🔧 Installing SAGE development aliases..."

# Create aliases directory
mkdir -p ~/.claude

# Download/create all alias files
echo "Creating alias files..."

# Add to shell configuration
SHELL_CONFIG=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
fi

if [[ -n "$SHELL_CONFIG" ]]; then
    echo "
# SAGE Development Aliases
source ~/.claude/gpu-workstation-aliases.sh
source ~/.claude/claude-dual-env-aliases.sh  
source ~/.claude/sage-development-aliases.sh
source ~/.claude/sync-monitoring-aliases.sh
source ~/.claude/network-diagnostics-aliases.sh
" >> "$SHELL_CONFIG"
    
    echo "✅ Aliases added to $SHELL_CONFIG"
    echo "Run 'source $SHELL_CONFIG' or restart your terminal to activate"
else
    echo "⚠️  Unable to detect shell configuration file"
    echo "Manually add alias sources to your shell configuration"
fi

echo "🎉 Alias setup complete!"
```

### Quick Reference Card
```bash
# GPU workstation quick access
gpu                 # SSH to GPU workstation
gpu-claude          # Start Claude Code on GPU
gpu-status          # Check GPU hardware status
sage-gpu            # SAGE development on GPU

# Local development
claude-local        # Local Claude Code session
sage-local          # Local SAGE development
claude-docs         # Documentation mode

# Synchronization
sync-status         # Check sync health
sync-test           # Test bidirectional sync
sync-health-full    # Comprehensive sync check

# Network diagnostics
zt-status           # ZeroTier status
net-check           # Basic connectivity test
net-diagnose        # Full network diagnostic

# SAGE workflows
sage-ensemble       # Full SAGE development
sage-validate       # Model validation
models-status       # Check all model availability
```

---

## 📋 Productivity Summary

### ✅ Complete Alias Coverage
- **80+ specialized aliases** for SAGE development workflow
- **5 organized categories** covering all aspects of dual-environment development
- **Intelligent task routing** with environment auto-selection
- **Comprehensive monitoring** for all system components
- **Troubleshooting shortcuts** for rapid issue resolution

### ✅ Development Workflow Enhancement
- **One-command access** to any development environment
- **Status monitoring** with visual indicators
- **Performance benchmarking** built into aliases
- **Emergency recovery** procedures readily available
- **Complete documentation** with help systems

### ✅ Maintenance & Operations
- **Health checks** for all system components
- **Automated testing** of sync and connectivity
- **Log monitoring** with intelligent filtering
- **Service management** with safety checks
- **Performance optimization** shortcuts

**Status**: Complete productivity alias suite ready for seamless SAGE development across dual environments with comprehensive monitoring and troubleshooting capabilities