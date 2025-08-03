# 🔄 SAGE-Forge Milestone Restoration Guide

**Complete instructions for restoring to any milestone**

## 🚨 Emergency Quick Restore

If you need to quickly restore to a working state:

```bash
# 1. Go to workspace root
cd /Users/terryli/eon/nt

# 2. List available milestones  
python milestones/milestone_manager.py list

# 3. Restore to latest working milestone
python milestones/milestone_manager.py restore sage-forge-gpu-ready --yes

# 4. Verify restoration
cd sage-forge-professional
uv run python tests/test_professional_structure.py
```

## 📋 Step-by-Step Restoration Process

### 1. Preparation
```bash
# Save your current work first!
git add .
git commit -m "WIP: saving before milestone restore"

# Navigate to workspace
cd /Users/terryli/eon/nt
```

### 2. Choose Milestone
```bash
# See all available milestones
python milestones/milestone_manager.py list

# Check milestone details
cat milestones/archives/[milestone-name]/MILESTONE.md
```

### 3. Automated Restoration
```bash
# Automated restore (recommended)
python milestones/milestone_manager.py restore [milestone-name]

# Force restore without confirmation
python milestones/milestone_manager.py restore [milestone-name] --yes
```

### 4. Manual Restoration (if automated fails)
```bash
# 1. Find the commit SHA
cat milestones/archives/[milestone-name]/milestone_metadata.json

# 2. Backup current state
git stash push -m "backup-$(date +%Y%m%d-%H%M%S)"

# 3. Restore to milestone commit
git checkout [COMMIT_SHA]

# 4. Create working branch
git checkout -b restore-[milestone-name]-$(date +%Y%m%d)
```

## 🔧 Verification Steps

After any restoration, verify the system works:

### Core System Check
```bash
cd sage-forge-professional

# 1. Test structure
uv run python tests/test_professional_structure.py
# Expected: 6/6 tests passed

# 2. Test ultimate demo  
uv run python demos/ultimate_complete_demo.py
# Expected: 214 orders executed successfully

# 3. Test CLI tools
./cli/sage-validate --all
```

### GPU Environment Check (if applicable)
```bash
# On GPU workstation
ssh zerotier-remote "cd ~/eon/nt/sage-forge-professional && source .venv-gpu/bin/activate && python3 -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\")'"
```

### Data Quality Check
```bash
# Run data quality diagnostics
uv run python debug/data_quality_diagnostics.py
# Expected: 100% core OHLCV data quality
```

## 🆘 Troubleshooting

### Common Issues

**Problem**: Git conflicts during restore
```bash
# Solution: Reset to clean state
git reset --hard HEAD
git clean -fd
# Then retry restoration
```

**Problem**: Missing dependencies after restore
```bash
# Solution: Reinstall environment
cd sage-forge-professional
uv sync
```

**Problem**: GPU environment not working
```bash
# Solution: Re-sync to GPU workstation
sage-sync --push-workspace --sync-sessions
```

**Problem**: Data quality validation fails
```bash
# Solution: Clear cache and retry
rm -rf data_cache/*
uv run python demos/ultimate_complete_demo.py
```

### Recovery Options

**Option 1**: Restore from backup
```bash
# List available stash backups
git stash list

# Restore specific backup
git stash apply stash@{0}
```

**Option 2**: Return to latest commit
```bash
# Go back to latest development
git checkout master
git pull  # if remote exists
```

**Option 3**: Emergency reset to known good state
```bash
# Reset to last known working tag
git tag --list | grep milestone | sort | tail -1
git checkout [LATEST_MILESTONE_TAG]
```

## 🌍 Remote Environment Restoration

### GPU Workstation Sync
```bash
# 1. Restore locally first
python milestones/milestone_manager.py restore [milestone-name]

# 2. Sync to GPU workstation
sage-sync --push-workspace --sync-sessions

# 3. Verify on remote
ssh zerotier-remote "cd ~/eon/nt && git log --oneline -1"
```

### Manual Remote Restoration
```bash
# If sage-sync not available
ssh zerotier-remote "cd ~/eon/nt && git checkout [COMMIT_SHA]"

# Re-setup GPU environment if needed
ssh zerotier-remote "cd ~/eon/nt/sage-forge-professional && source .venv-gpu/bin/activate && pip install --index-url https://pypi.org/simple/ torch torchvision torchaudio --upgrade"
```

## 📊 Validation Checklist

After restoration, check all these items:

### ✅ File Structure
- [ ] `sage-forge-professional/` directory exists
- [ ] `src/sage_forge/` framework is complete  
- [ ] `cli/` tools are executable
- [ ] `configs/` files are present
- [ ] `tests/` directory is complete
- [ ] `demos/ultimate_complete_demo.py` exists (33KB+)

### ✅ Functionality
- [ ] Professional structure tests pass (6/6)
- [ ] Ultimate demo executes successfully (214 orders)
- [ ] Data quality validation achieves 100%
- [ ] CLI tools respond correctly
- [ ] GPU environment works (if applicable)

### ✅ Development Environment
- [ ] All imports work without errors
- [ ] Configuration files load properly
- [ ] Documentation is accessible
- [ ] Git history is intact
- [ ] Remote sync works

## 🔐 Safety Guidelines

1. **Always backup before restoring**: Use git stash or commits
2. **Verify before continuing**: Run all validation checks
3. **Document issues**: Note any problems in milestone docs
4. **Test in isolation**: Verify restoration doesn't break other systems
5. **Sync remote environments**: Ensure GPU workstation matches

## 📞 Emergency Contacts

**If restoration fails completely**:
1. Check `git reflog` for recent commits
2. Look for backup stashes: `git stash list`
3. Check investigation archives: `/Users/terryli/eon/nt/investigation_archive/`
4. Review milestone archives: `/Users/terryli/eon/nt/milestones/archives/`

---

**Last Updated**: 2025-08-02  
**Version**: 1.0  
**Tested Restorations**: 0 (system newly created)