#!/bin/bash
# 🔄 SAGE-Forge Milestone Restoration Script
# Milestone: 2025-08-03-tirex-nx-ai-model-integration-complete
# Created: 2025-08-03 03:05:40

set -e

echo "🔄 Restoring SAGE-Forge to milestone: 2025-08-03-tirex-nx-ai-model-integration-complete"
echo "📝 Target commit: c5d91f2ee1f1f49c983a93646067acccc3e717b5"

# Backup current state
BACKUP_NAME="backup-before-restore-$(date +%Y%m%d-%H%M%S)"
echo "💾 Creating backup: $BACKUP_NAME"
git stash push -m "$BACKUP_NAME" || true

# Restore to milestone commit
echo "🔄 Checking out commit c5d91f2ee1f1f49c983a93646067acccc3e717b5..."
git checkout c5d91f2ee1f1f49c983a93646067acccc3e717b5

# Verify restoration
echo "🧪 Verifying restoration..."
if [ -f "sage-forge-professional/tests/test_professional_structure.py" ]; then
    cd sage-forge-professional
    uv run python tests/test_professional_structure.py
    cd ..
fi

echo "✅ Milestone 2025-08-03-tirex-nx-ai-model-integration-complete restored successfully"
echo "💾 Previous state backed up as: $BACKUP_NAME"
echo ""
echo "🚀 Next steps:"
echo "  - Verify functionality: cd sage-forge-professional && uv run python demos/ultimate_complete_demo.py"
echo "  - Continue development: git checkout -b continue-from-2025-08-03-tirex-nx-ai-model-integration-complete"
echo "  - Restore backup if needed: git stash pop"
