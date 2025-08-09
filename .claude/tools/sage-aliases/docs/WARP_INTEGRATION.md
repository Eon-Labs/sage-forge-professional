# Warp Terminal Integration for GPU Workstation

## Warp Subshell Configuration

### 1. Add Custom Subshell Command

In Warp Settings → Features → Subshells, add:

```
gpu-ws
```

### 2. Remote Shell RC Configuration (Optional)

Add to `~/.bashrc` or `~/.zshrc` on GPU workstation for auto-warpify:

```bash
# Warp subshell integration
if [[ "$TERM_PROGRAM" == "WarpTerminal" ]] && [[ -z "$WARP_IS_LOCAL_SHELL_SESSION" ]]; then
    export WARP_IS_LOCAL_SHELL_SESSION="false"
fi
```

### 3. Optimized Usage Patterns

#### Interactive Development Session (Warp Subshell)

```bash
gpu-ws                    # Pure SSH session - Warp will warpify
gpu-ws claude            # Direct Claude Code session
gpu-ws sage              # SAGE development with GPU status
```

#### Quick Commands (Non-Interactive)

```bash
gpu-ws status            # GPU status check
gpu-ws check             # Connectivity test
```

## Warp Features Available in GPU Subshell

- **AI Command Suggestions**: Works in SSH sessions
- **Command History**: Synced across sessions
- **Block Output**: Organized command/output blocks
- **Workflows**: Can save GPU development workflows
- **Background Commands**: Handled during idle time

## Troubleshooting

### If Subshell Not Detected

1. Ensure remote shell is bash/zsh/fish
2. Check Warp settings include `gpu-ws`
3. Verify `-t` flag in SSH command (already included)

### Performance Tips

- Use `gpu-ws sage` for full development setup
- Background commands may run during idle time
- Disable background commands if causing issues: Warp Settings → Features → Subshells → "Run background commands in subshells"
