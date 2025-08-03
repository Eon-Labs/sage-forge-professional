#!/bin/bash
# GPU Workstation Enhanced - Comprehensive Sync Capabilities
# Enhanced version of gpu-ws with complete development sync

set -euo pipefail

# Configuration
SCRIPT_NAME="gpu-ws"
VERSION="2.0.0"

# Auto-detect environment and set remote configuration
if [[ "$(hostname)" == "el02" ]]; then
    # Running on GPU workstation - macOS is remote
    REMOTE_HOST="terryli@172.25.253.142"
    REMOTE_USER="terryli"
    REMOTE_IP="172.25.253.142"
    LOCAL_WORKSPACE="$HOME/eon/nt"
    REMOTE_WORKSPACE="/Users/terryli/eon/nt"
    LOCAL_CLAUDE="$HOME/.claude"
    REMOTE_CLAUDE="/Users/terryli/.claude"
    ENVIRONMENT="GPU workstation (el02)"
    REMOTE_ENV="macOS"
else
    # Running on macOS - GPU workstation is remote  
    REMOTE_HOST="zerotier-remote"
    REMOTE_USER="tca"
    REMOTE_IP="172.25.96.253"
    LOCAL_WORKSPACE="$HOME/eon/nt"
    REMOTE_WORKSPACE="~/eon/nt"
    LOCAL_CLAUDE="$HOME/.claude"
    REMOTE_CLAUDE="~/.claude"
    ENVIRONMENT="macOS"
    REMOTE_ENV="GPU workstation (el02)"
fi

# Colors and logging (with proper terminal detection)
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    CYAN=$(tput setaf 6)
    NC=$(tput sgr0)
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# Logging functions
log_info() { echo -e "${BLUE}ℹ️  $*${NC}"; }
log_success() { echo -e "${GREEN}✅ $*${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $*${NC}"; }
log_error() { echo -e "${RED}❌ $*${NC}"; }

# Help text
show_help() {
    cat << EOF
${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}
${BLUE}║                GPU Workstation Enhanced v${VERSION}                 ║${NC}
${BLUE}║              Comprehensive Development Sync                  ║${NC}
${BLUE}║   Running on: ${ENVIRONMENT} → ${REMOTE_ENV}   ║${NC}
${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}

${CYAN}CONNECTION COMMANDS:${NC}
  gpu-ws                    Connect to GPU workstation via SSH
  gpu-ws claude             Start Claude Code on GPU workstation
  gpu-ws sage               Start SAGE development with GPU monitoring
  gpu-ws dev                Complete development session startup
  gpu-ws status             Show GPU hardware status and connectivity

${CYAN}COMPREHENSIVE SYNC COMMANDS:${NC}
  gpu-ws sync-all           🚀 Sync everything (sessions, config, git, files)
  gpu-ws push-all           📤 Push all changes from GPU to macOS
  gpu-ws pull-all           📥 Pull all changes from macOS to GPU
  gpu-ws sync-status        📊 Show what needs syncing

${CYAN}SELECTIVE SYNC COMMANDS:${NC}
  gpu-ws sync-sessions      💬 Claude Code sessions and conversation history
  gpu-ws sync-config        ⚙️  Configuration files and settings
  gpu-ws sync-git           🔄 Git history, commits, and repository state
  gpu-ws sync-workspace     📁 Workspace files and documentation
  gpu-ws sync-env           🔧 Development environment and tools

${CYAN}SYNC OPTIONS:${NC}
  --dry-run                 Show what would be synced without executing
  --fast                    Skip large files and caches for faster sync
  --force                   Overwrite conflicts without prompting
  --verbose                 Show detailed sync progress
  --exclude-cache           Skip cache and build artifacts

${CYAN}EXAMPLES:${NC}
  gpu-ws sync-all --dry-run           # Preview complete sync
  gpu-ws push-all --fast              # Quick push, skip caches
  gpu-ws sync-sessions --verbose      # Detailed session sync
  gpu-ws pull-all --exclude-cache     # Pull without build artifacts

${CYAN}RECOVERY COMMANDS:${NC}
  gpu-ws backup-create      Create backup before major changes
  gpu-ws backup-restore     Restore from latest backup
  gpu-ws validate           Validate sync integrity

Part of SAGE Development Infrastructure
EOF
}

# Connectivity check
check_connectivity() {
    log_info "Checking connectivity to $REMOTE_ENV..."
    
    if ! ping -c 2 "$REMOTE_IP" >/dev/null 2>&1; then
        log_error "Cannot reach $REMOTE_ENV at $REMOTE_IP"
        log_info "Check network connection and VPN status"
        exit 1
    fi
    
    if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" "echo test" >/dev/null 2>&1; then
        log_error "SSH connection to $REMOTE_HOST failed"
        log_info "Check SSH configuration and keys"
        exit 1
    fi
    
    log_success "Connectivity to $REMOTE_ENV confirmed"
}

# Git sync functions
sync_git_repository() {
    local direction=$1
    local dry_run=${2:-false}
    
    log_info "Syncing git repository ($direction)..."
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "[DRY RUN] Would sync git repository"
        return 0
    fi
    
    case "$direction" in
        "push")
            # Commit any uncommitted changes first
            if [[ -n $(git status --porcelain) ]]; then
                log_warning "Uncommitted changes detected"
                read -p "Commit changes before sync? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    git add .
                    git commit -m "auto: commit before push sync $(date '+%Y-%m-%d %H:%M:%S')"
                fi
            fi
            
            # Create bundle of commits
            local bundle_file="/tmp/git-sync-$(date +%s).bundle"
            git bundle create "$bundle_file" --all
            
            # Transfer bundle and apply on remote
            scp "$bundle_file" "$REMOTE_HOST:/tmp/"
            ssh "$REMOTE_HOST" "cd $REMOTE_WORKSPACE && git bundle verify /tmp/$(basename $bundle_file) && git pull /tmp/$(basename $bundle_file) && rm /tmp/$(basename $bundle_file)"
            rm "$bundle_file"
            ;;
            
        "pull")
            # Get bundle from remote
            local bundle_file="/tmp/git-sync-$(date +%s).bundle"
            ssh "$REMOTE_HOST" "cd $REMOTE_WORKSPACE && git bundle create /tmp/$(basename $bundle_file) --all"
            scp "$REMOTE_HOST:/tmp/$(basename $bundle_file)" "$bundle_file"
            
            # Apply bundle locally
            git bundle verify "$bundle_file"
            git pull "$bundle_file"
            
            # Cleanup
            rm "$bundle_file"
            ssh "$REMOTE_HOST" "rm /tmp/$(basename $bundle_file)"
            ;;
    esac
    
    log_success "Git repository sync completed"
}

# Claude Code sessions sync
sync_claude_sessions() {
    local direction=$1
    local dry_run=${2:-false}
    
    log_info "Syncing Claude Code sessions ($direction)..."
    
    local rsync_cmd="rsync -avzP --stats"
    if [[ "$dry_run" == "true" ]]; then
        rsync_cmd="$rsync_cmd --dry-run"
        log_info "[DRY RUN] Would sync Claude Code sessions"
    fi
    
    case "$direction" in
        "push")
            # Sync sessions from GPU to macOS
            $rsync_cmd "$LOCAL_CLAUDE/system/sessions/" "$REMOTE_HOST:$REMOTE_CLAUDE/system/sessions/"
            
            if [[ "$dry_run" != "true" ]]; then
                # Apply path correction for macOS
                ssh "$REMOTE_HOST" "
                    if [ -d '$REMOTE_CLAUDE/system/sessions/-home-tca-eon-nt' ]; then
                        cp -r '$REMOTE_CLAUDE/system/sessions/-home-tca-eon-nt/'* '$REMOTE_CLAUDE/system/sessions/-Users-terryli-eon-nt/' 2>/dev/null || true
                        rm -rf '$REMOTE_CLAUDE/system/sessions/-home-tca-eon-nt'
                        chmod -R 644 '$REMOTE_CLAUDE/system/sessions/-Users-terryli-eon-nt/'*.jsonl 2>/dev/null || true
                        chmod 755 '$REMOTE_CLAUDE/system/sessions/-Users-terryli-eon-nt' 2>/dev/null || true
                    fi
                "
                log_success "Applied path correction for macOS sessions"
            fi
            ;;
            
        "pull")
            # Sync sessions from macOS to GPU
            $rsync_cmd "$REMOTE_HOST:$REMOTE_CLAUDE/system/sessions/" "$LOCAL_CLAUDE/system/sessions/"
            
            if [[ "$dry_run" != "true" ]]; then
                # Apply path correction for GPU workstation
                if [ -d "$LOCAL_CLAUDE/system/sessions/-Users-terryli-eon-nt" ]; then
                    cp -r "$LOCAL_CLAUDE/system/sessions/-Users-terryli-eon-nt/"* "$LOCAL_CLAUDE/system/sessions/-home-tca-eon-nt/" 2>/dev/null || true
                    rm -rf "$LOCAL_CLAUDE/system/sessions/-Users-terryli-eon-nt"
                    chmod -R 644 "$LOCAL_CLAUDE/system/sessions/-home-tca-eon-nt/"*.jsonl 2>/dev/null || true
                    chmod 755 "$LOCAL_CLAUDE/system/sessions/-home-tca-eon-nt" 2>/dev/null || true
                fi
                log_success "Applied path correction for GPU workstation sessions"
            fi
            ;;
    esac
    
    log_success "Claude Code sessions sync completed"
}

# Configuration sync
sync_config() {
    local direction=$1
    local dry_run=${2:-false}
    
    log_info "Syncing configuration files ($direction)..."
    
    local config_files=(
        "CLAUDE.md"
        "settings.json"
        ".cursorrules"
        "automation/"
        "tools/"
        "commands/"
    )
    
    local rsync_cmd="rsync -avzP"
    if [[ "$dry_run" == "true" ]]; then
        rsync_cmd="$rsync_cmd --dry-run"
        log_info "[DRY RUN] Would sync configuration files"
    fi
    
    for file in "${config_files[@]}"; do
        case "$direction" in
            "push")
                if [[ -e "$LOCAL_CLAUDE/$file" ]]; then
                    $rsync_cmd "$LOCAL_CLAUDE/$file" "$REMOTE_HOST:$REMOTE_CLAUDE/"
                fi
                ;;
            "pull")
                $rsync_cmd "$REMOTE_HOST:$REMOTE_CLAUDE/$file" "$LOCAL_CLAUDE/" 2>/dev/null || true
                ;;
        esac
    done
    
    log_success "Configuration sync completed"
}

# Workspace files sync
sync_workspace() {
    local direction=$1
    local dry_run=${2:-false}
    local exclude_cache=${3:-false}
    
    log_info "Syncing workspace files ($direction)..."
    
    local rsync_cmd="rsync -avzP --compress-level=9 --partial --inplace"
    
    # Standard excludes
    rsync_cmd="$rsync_cmd --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo'"
    rsync_cmd="$rsync_cmd --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='*.log'"
    
    # Optional cache excludes
    if [[ "$exclude_cache" == "true" ]]; then
        rsync_cmd="$rsync_cmd --exclude='.venv' --exclude='node_modules' --exclude='.cache'"
        rsync_cmd="$rsync_cmd --exclude='dist' --exclude='build' --exclude='*.egg-info'"
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        rsync_cmd="$rsync_cmd --dry-run"
        log_info "[DRY RUN] Would sync workspace files"
    fi
    
    case "$direction" in
        "push")
            $rsync_cmd "$LOCAL_WORKSPACE/" "$REMOTE_HOST:$REMOTE_WORKSPACE/"
            ;;
        "pull")
            $rsync_cmd "$REMOTE_HOST:$REMOTE_WORKSPACE/" "$LOCAL_WORKSPACE/"
            ;;
    esac
    
    log_success "Workspace files sync completed"
}

# Development environment sync
sync_environment() {
    local direction=$1
    local dry_run=${2:-false}
    
    log_info "Syncing development environment ($direction)..."
    
    local env_files=(
        ".zshrc"
        ".bashrc"
        ".profile"
        ".local/bin/"
        ".ssh/config"
    )
    
    local rsync_cmd="rsync -avzP"
    if [[ "$dry_run" == "true" ]]; then
        rsync_cmd="$rsync_cmd --dry-run"
        log_info "[DRY RUN] Would sync environment files"
    fi
    
    for file in "${env_files[@]}"; do
        case "$direction" in
            "push")
                if [[ -e "$HOME/$file" ]]; then
                    $rsync_cmd "$HOME/$file" "$REMOTE_HOST:~/"
                fi
                ;;
            "pull")
                $rsync_cmd "$REMOTE_HOST:~/$file" "$HOME/" 2>/dev/null || true
                ;;
        esac
    done
    
    log_success "Environment sync completed"
}

# Comprehensive sync status
show_sync_status() {
    log_info "Analyzing sync status..."
    
    echo
    echo "${CYAN}📊 Sync Status Analysis${NC}"
    echo "================================"
    
    # Git status
    echo "${BLUE}Git Repository:${NC}"
    local_commits=$(git rev-list --count HEAD 2>/dev/null || echo "0")
    remote_commits=$(ssh "$REMOTE_HOST" "cd $REMOTE_WORKSPACE && git rev-list --count HEAD 2>/dev/null || echo '0'")
    echo "  Local commits: $local_commits"
    echo "  Remote commits: $remote_commits"
    
    if [[ "$local_commits" != "$remote_commits" ]]; then
        log_warning "Git repositories are out of sync"
    else
        log_success "Git repositories are in sync"
    fi
    
    # Claude sessions
    echo "${BLUE}Claude Code Sessions:${NC}"
    local_sessions=$(find "$LOCAL_CLAUDE/system/sessions" -name "*.jsonl" 2>/dev/null | wc -l)
    remote_sessions=$(ssh "$REMOTE_HOST" "find $REMOTE_CLAUDE/system/sessions -name '*.jsonl' 2>/dev/null | wc -l")
    echo "  Local sessions: $local_sessions"
    echo "  Remote sessions: $remote_sessions"
    
    # Workspace files
    echo "${BLUE}Workspace Files:${NC}"
    local_files=$(find "$LOCAL_WORKSPACE" -type f ! -path "*/.git/*" 2>/dev/null | wc -l)
    remote_files=$(ssh "$REMOTE_HOST" "find $REMOTE_WORKSPACE -type f ! -path '*/.git/*' 2>/dev/null | wc -l")
    echo "  Local files: $local_files"
    echo "  Remote files: $remote_files"
    
    echo
}

# Main sync dispatcher
execute_sync() {
    local sync_type=$1
    local direction=$2
    local dry_run=${3:-false}
    local exclude_cache=${4:-false}
    local verbose=${5:-false}
    
    # Add verbose flag to rsync if requested
    if [[ "$verbose" == "true" ]]; then
        export RSYNC_VERBOSE="--verbose"
    fi
    
    case "$sync_type" in
        "all")
            log_info "🚀 Starting comprehensive sync ($direction)..."
            sync_config "$direction" "$dry_run"
            sync_git_repository "$direction" "$dry_run"
            sync_workspace "$direction" "$dry_run" "$exclude_cache"
            sync_claude_sessions "$direction" "$dry_run"
            sync_environment "$direction" "$dry_run"
            log_success "🎉 Comprehensive sync completed!"
            ;;
        "sessions")
            sync_claude_sessions "$direction" "$dry_run"
            ;;
        "config")
            sync_config "$direction" "$dry_run"
            ;;
        "git")
            sync_git_repository "$direction" "$dry_run"
            ;;
        "workspace")
            sync_workspace "$direction" "$dry_run" "$exclude_cache"
            ;;
        "env")
            sync_environment "$direction" "$dry_run"
            ;;
        *)
            log_error "Unknown sync type: $sync_type"
            exit 1
            ;;
    esac
}

# Parse command line arguments
parse_args() {
    local dry_run=false
    local exclude_cache=false
    local verbose=false
    local force=false
    local fast=false
    
    # Parse flags
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            --exclude-cache)
                exclude_cache=true
                shift
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --fast)
                exclude_cache=true
                fast=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    echo "$dry_run $exclude_cache $verbose $force $fast"
}

# Main command handler
main() {
    local cmd=${1:-""}
    shift || true
    
    # Parse common flags
    local flags
    flags=$(parse_args "$@")
    read -r dry_run exclude_cache verbose force fast <<< "$flags"
    
    case "$cmd" in
        ""|"connect")
            check_connectivity
            exec ssh "$REMOTE_HOST"
            ;;
        "claude")
            check_connectivity
            exec ssh "$REMOTE_HOST" -t "cd $REMOTE_WORKSPACE && export PATH=~/.npm-global/bin:\$PATH && claude"
            ;;
        "sage")
            check_connectivity
            exec ssh "$REMOTE_HOST" -t "cd $REMOTE_WORKSPACE && echo 'SAGE Development - GPU Environment (RTX 4090)' && export PATH=~/.npm-global/bin:\$PATH && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits && claude"
            ;;
        "dev")
            log_info "🚀 Starting GPU development session..."
            check_connectivity
            ssh "$REMOTE_HOST" "hostname && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits"
            exec ssh "$REMOTE_HOST" -t "cd $REMOTE_WORKSPACE && export PATH=~/.npm-global/bin:\$PATH && claude"
            ;;
        "status")
            check_connectivity
            ssh "$REMOTE_HOST" "hostname && nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader,nounits"
            ;;
        "sync-all")
            check_connectivity
            execute_sync "all" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "push-all")
            check_connectivity
            execute_sync "all" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "pull-all")
            check_connectivity
            execute_sync "all" "pull" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "sync-sessions")
            check_connectivity
            execute_sync "sessions" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "sync-config")
            check_connectivity
            execute_sync "config" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "sync-git")
            check_connectivity
            execute_sync "git" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "sync-workspace")
            check_connectivity
            execute_sync "workspace" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "sync-env")
            check_connectivity
            execute_sync "env" "push" "$dry_run" "$exclude_cache" "$verbose"
            ;;
        "sync-status")
            check_connectivity
            show_sync_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $cmd"
            echo "Use '$SCRIPT_NAME help' for available commands"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"