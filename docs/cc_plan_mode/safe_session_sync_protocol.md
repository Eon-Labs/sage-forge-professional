# 🛡️ Safe Claude Code Session Synchronization Protocol

**Based on research and known issues with Claude Code session conflicts**

## ⚠️ **Critical Safety Rules**

### **1. Session Corruption Risks**
- **Interrupted sessions** during tool execution cause permanent API sync errors
- **Multiple concurrent sessions** mix command history and contexts
- **Concurrent sync** can corrupt session files and lose data permanently

### **2. Pre-Sync Safety Checklist**

#### **Step 1: Check for Active Claude Sessions**

**Local macOS Check:**
```bash
# Check for Claude Code CLI processes
ps aux | grep -E "(claude|Claude)" | grep -v grep

# Check for specific Claude Code node processes
ps aux | grep "@anthropic-ai/claude-code" | grep -v grep

# Quick check
pgrep -fl "claude"
```

**GPU Workstation Check:**
```bash
ssh zerotier-remote "export PATH=\"~/.local/bin:\$PATH\" && ps aux | grep -E '(claude|Claude)' | grep -v grep"
```

#### **Step 2: Graceful Session Termination**
```bash
# If active sessions found, terminate gracefully:
# 1. Save work in progress in the Claude session
# 2. Exit Claude sessions with Ctrl+D (not Ctrl+C)
# 3. Wait for clean shutdown
# 4. Verify no processes remain
```

#### **Step 3: Session File Lock Check**
```bash
# Check for session lock files (if any)
ls -la ~/.claude/system/sessions/.lock* 2>/dev/null || echo "No lock files"

# Check for temporary session files
ls -la ~/.claude/system/sessions/tmp* 2>/dev/null || echo "No temp files"
```

### **3. Safe Sync Commands**

#### **One-Way Sync (macOS → GPU Workstation)**
```bash
# Sync current TiRex session
rsync -avz --progress ~/.claude/system/sessions/-Users-terryli-eon-nt/ \
    zerotier-remote:~/.claude/system/sessions/-home-tca-eon-nt/

# Sync all sessions (if needed)
rsync -avz --progress ~/.claude/system/sessions/ \
    zerotier-remote:~/.claude/system/sessions/
```

#### **Bi-Directional Sync (Use with EXTREME caution)**
```bash
# Only if you need to sync changes back from GPU workstation
# NEVER do this if both sides have been active simultaneously

# GPU → Local (after verifying no conflicts)
rsync -avz --progress zerotier-remote:~/.claude/system/sessions/-home-tca-eon-nt/ \
    ~/.claude/system/sessions/-Users-terryli-eon-nt/
```

### **4. Post-Sync Verification**

```bash
# Verify file counts match
echo "Local files:" && ls ~/.claude/system/sessions/-Users-terryli-eon-nt/ | wc -l
echo "Remote files:" && ssh zerotier-remote "ls ~/.claude/system/sessions/-home-tca-eon-nt/ | wc -l"

# Verify largest session file integrity
ls -lah ~/.claude/system/sessions/-Users-terryli-eon-nt/ | head -5
ssh zerotier-remote "ls -lah ~/.claude/system/sessions/-home-tca-eon-nt/ | head -5"
```

## 🚨 **Emergency Recovery**

### **If Sessions Get Corrupted**
1. **Stop all Claude processes immediately**
2. **Backup corrupted sessions**: `cp -r ~/.claude/system/sessions/ ~/.claude/sessions_backup_$(date +%s)/`
3. **Restore from last known good state** (use milestone system)
4. **Restart Claude and verify functionality**

### **If Sync Conflicts Occur**
1. **Identify conflict source**: Compare file timestamps and sizes
2. **Choose authoritative source**: Usually the side with most recent meaningful work
3. **Manually resolve**: Copy specific session files rather than bulk sync
4. **Test session integrity** before continuing work

## 🔧 **Automation Scripts**

### **Safe Sync Script**
```bash
#!/bin/bash
# safe-claude-sync.sh

echo "🔍 Checking for active Claude sessions..."

# Check local
if pgrep -f "claude" > /dev/null; then
    echo "❌ Active Claude session found locally. Please exit Claude first."
    exit 1
fi

# Check remote
if ssh zerotier-remote "pgrep -f claude" > /dev/null 2>&1; then
    echo "❌ Active Claude session found on GPU workstation. Please exit Claude first."
    exit 1
fi

echo "✅ No active sessions found. Safe to sync."

# Perform sync
rsync -avz --progress ~/.claude/system/sessions/-Users-terryli-eon-nt/ \
    zerotier-remote:~/.claude/system/sessions/-home-tca-eon-nt/

echo "✅ Sync completed successfully."
```

## 📋 **Best Practices**

1. **Always exit Claude gracefully** (Ctrl+D, not Ctrl+C)
2. **Sync during natural break points** (end of tasks, before switching machines)
3. **Use milestone system** for major sync points
4. **Keep sync direction consistent** (prefer one-way: macOS → GPU)
5. **Verify after every sync** before starting new sessions
6. **Backup before major syncs** using milestone system

## ⚡ **Quick Reference**

**Before Sync**: Check processes on both sides  
**During Sync**: Use rsync with progress monitoring  
**After Sync**: Verify file counts and start fresh sessions  
**If Problems**: Stop everything, backup, restore from milestone  

---

**Remember**: Session corruption can cause permanent data loss. Always follow the safety protocol!