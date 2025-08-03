#!/bin/bash
# 🔄 SAGE-Forge Milestone Restoration Script
# Milestone: 2025-08-02-sage-forge-gpu-ready
# Created: 2025-08-02 21:31:57

set -e

echo "🔄 Restoring SAGE-Forge to milestone: 2025-08-02-sage-forge-gpu-ready"
echo "📝 Target commit: 78c29dbd3902428f02142966aa6b5949b685a782"

# Backup current state
BACKUP_NAME="backup-before-restore-$(date +%Y%m%d-%H%M%S)"
echo "💾 Creating backup: $BACKUP_NAME"
git stash push -m "$BACKUP_NAME" || true

# Restore to milestone commit
echo "🔄 Checking out commit 78c29dbd3902428f02142966aa6b5949b685a782..."
git checkout 78c29dbd3902428f02142966aa6b5949b685a782

# Verify restoration
echo "🧪 Verifying restoration..."
if [ -f "sage-forge-professional/tests/test_professional_structure.py" ]; then
    cd sage-forge-professional
    uv run python tests/test_professional_structure.py
    cd ..
fi

echo "✅ Milestone 2025-08-02-sage-forge-gpu-ready restored successfully"
echo "💾 Previous state backed up as: $BACKUP_NAME"
echo ""
echo "🚀 Next steps:"
echo "  - Verify functionality: cd sage-forge-professional && uv run python demos/ultimate_complete_demo.py"
echo "  - Continue development: git checkout -b continue-from-2025-08-02-sage-forge-gpu-ready"
echo "  - Restore backup if needed: git stash pop"
