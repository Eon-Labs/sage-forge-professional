#!/usr/bin/env python3
"""
🏆 SAGE-Forge Milestone Manager

Automated creation and restoration of development milestones with complete
workspace snapshots, git integration, and GPU workstation synchronization.
"""

import json
import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class MilestoneManager:
    """Comprehensive milestone management for SAGE-Forge development."""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.milestones_dir = self.workspace_root / "milestones"
        self.archives_dir = self.milestones_dir / "archives"
        self.index_file = self.milestones_dir / "current_milestones.json"
        
        # Ensure milestone directories exist
        self.milestones_dir.mkdir(exist_ok=True)
        self.archives_dir.mkdir(exist_ok=True)
        
    def create_milestone(self, description: str, force: bool = False) -> str:
        """Create a new milestone with complete workspace snapshot."""
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        milestone_name = f"{date_str}-{description.lower().replace(' ', '-')}"
        milestone_dir = self.archives_dir / milestone_name
        
        print(f"🏆 Creating milestone: {milestone_name}")
        
        # Check if milestone already exists
        if milestone_dir.exists() and not force:
            print(f"❌ Milestone {milestone_name} already exists. Use --force to overwrite.")
            return ""
            
        # Create git commit and tag
        commit_sha = self._create_git_milestone(milestone_name, description)
        if not commit_sha:
            print("❌ Failed to create git milestone")
            return ""
            
        # Create milestone directory
        if milestone_dir.exists():
            shutil.rmtree(milestone_dir)
        milestone_dir.mkdir(parents=True)
        
        # Create milestone documentation
        self._create_milestone_docs(milestone_dir, milestone_name, description, commit_sha)
        
        # Create workspace snapshot
        self._create_workspace_snapshot(milestone_dir)
        
        # Update milestone index
        self._update_milestone_index(milestone_name, description, commit_sha, timestamp)
        
        # Create restoration script
        self._create_restore_script(milestone_dir, milestone_name, commit_sha)
        
        print(f"✅ Milestone created: {milestone_name}")
        print(f"📁 Location: {milestone_dir}")
        print(f"🏷️ Git tag: milestone-{milestone_name}")
        print(f"📝 Commit: {commit_sha}")
        
        return milestone_name
        
    def list_milestones(self) -> List[Dict]:
        """List all available milestones."""
        if not self.index_file.exists():
            return []
            
        with open(self.index_file, 'r') as f:
            milestones = json.load(f)
            
        print("🏆 Available Milestones:")
        print("=" * 60)
        
        for milestone in sorted(milestones, key=lambda x: x['created'], reverse=True):
            print(f"📅 {milestone['name']}")
            print(f"   Description: {milestone['description']}")
            print(f"   Created: {milestone['created']}")
            print(f"   Commit: {milestone['commit_sha'][:8]}")
            print()
            
        return milestones
        
    def restore_milestone(self, milestone_name: str, confirm: bool = False) -> bool:
        """Restore workspace to specified milestone."""
        milestone_dir = self.archives_dir / milestone_name
        
        if not milestone_dir.exists():
            print(f"❌ Milestone {milestone_name} not found")
            return False
            
        if not confirm:
            response = input(f"⚠️ This will restore workspace to {milestone_name}. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Restoration cancelled")
                return False
                
        print(f"🔄 Restoring to milestone: {milestone_name}")
        
        # Read milestone metadata
        metadata_file = milestone_dir / "milestone_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                commit_sha = metadata['commit_sha']
        else:
            print("❌ Milestone metadata not found")
            return False
            
        # Backup current state
        backup_name = f"backup-before-restore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._backup_current_state(backup_name)
        
        # Restore git state
        try:
            subprocess.run(["git", "checkout", commit_sha], check=True, cwd=self.workspace_root)
            print(f"✅ Git restored to commit {commit_sha[:8]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Git restoration failed: {e}")
            return False
            
        # Verify restoration
        if self._verify_restoration(milestone_dir):
            print(f"✅ Milestone {milestone_name} restored successfully")
            print(f"💾 Backup saved as: {backup_name}")
            return True
        else:
            print(f"⚠️ Restoration completed but verification failed")
            return False
            
    def _create_git_milestone(self, milestone_name: str, description: str) -> str:
        """Create git commit and tag for milestone."""
        try:
            # Stage all changes
            subprocess.run(["git", "add", "."], check=True, cwd=self.workspace_root)
            
            # Create commit
            commit_msg = f"milestone: {description}\n\nMilestone: {milestone_name}\nCreated: {datetime.now().isoformat()}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, cwd=self.workspace_root)
            
            # Get commit SHA
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=self.workspace_root)
            commit_sha = result.stdout.strip()
            
            # Create tag
            tag_name = f"milestone-{milestone_name}"
            subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Milestone: {description}"], check=True, cwd=self.workspace_root)
            
            print(f"🏷️ Created git tag: {tag_name}")
            return commit_sha
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return ""
            
    def _create_milestone_docs(self, milestone_dir: Path, milestone_name: str, description: str, commit_sha: str):
        """Create milestone documentation from template."""
        template_file = self.milestones_dir / "TEMPLATE_MILESTONE.md"
        milestone_doc = milestone_dir / "MILESTONE.md"
        
        if template_file.exists():
            with open(template_file, 'r') as f:
                template_content = f.read()
                
            # Replace template placeholders
            content = template_content.replace("[MILESTONE_NAME]", milestone_name)
            content = content.replace("[YYYY-MM-DD]", datetime.now().strftime("%Y-%m-%d"))
            content = content.replace("[description]", description)
            content = content.replace("[COMMIT_SHA]", commit_sha)
            content = content.replace("[YYYY-MM-DD HH:MM:SS]", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            with open(milestone_doc, 'w') as f:
                f.write(content)
                
        # Create metadata file
        metadata = {
            "name": milestone_name,
            "description": description,
            "created": datetime.now().isoformat(),
            "commit_sha": commit_sha,
            "git_tag": f"milestone-{milestone_name}",
            "workspace_root": str(self.workspace_root),
            "restoration_verified": False
        }
        
        with open(milestone_dir / "milestone_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _create_workspace_snapshot(self, milestone_dir: Path):
        """Create snapshot of critical workspace files."""
        snapshot_dir = milestone_dir / "workspace_snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        
        # Critical directories to snapshot
        critical_dirs = [
            "sage-forge-professional/src",
            "sage-forge-professional/cli", 
            "sage-forge-professional/configs",
            "sage-forge-professional/tests",
            "sage-forge-professional/demos",
            "sage-forge-professional/documentation"
        ]
        
        for dir_name in critical_dirs:
            src_dir = self.workspace_root / dir_name
            if src_dir.exists():
                dst_dir = snapshot_dir / dir_name
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                
        # Critical files
        critical_files = [
            "sage-forge-professional/README.md",
            "sage-forge-professional/pyproject.toml",
            "sage-forge-professional/setup_sage_forge.py",
            "CLAUDE.md"
        ]
        
        for file_name in critical_files:
            src_file = self.workspace_root / file_name
            if src_file.exists():
                dst_file = snapshot_dir / file_name
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                
    def _create_restore_script(self, milestone_dir: Path, milestone_name: str, commit_sha: str):
        """Create automated restoration script."""
        script_content = f"""#!/bin/bash
# 🔄 SAGE-Forge Milestone Restoration Script
# Milestone: {milestone_name}
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

set -e

echo "🔄 Restoring SAGE-Forge to milestone: {milestone_name}"
echo "📝 Target commit: {commit_sha}"

# Backup current state
BACKUP_NAME="backup-before-restore-$(date +%Y%m%d-%H%M%S)"
echo "💾 Creating backup: $BACKUP_NAME"
git stash push -m "$BACKUP_NAME" || true

# Restore to milestone commit
echo "🔄 Checking out commit {commit_sha}..."
git checkout {commit_sha}

# Verify restoration
echo "🧪 Verifying restoration..."
if [ -f "sage-forge-professional/tests/test_professional_structure.py" ]; then
    cd sage-forge-professional
    uv run python tests/test_professional_structure.py
    cd ..
fi

echo "✅ Milestone {milestone_name} restored successfully"
echo "💾 Previous state backed up as: $BACKUP_NAME"
echo ""
echo "🚀 Next steps:"
echo "  - Verify functionality: cd sage-forge-professional && uv run python demos/ultimate_complete_demo.py"
echo "  - Continue development: git checkout -b continue-from-{milestone_name}"
echo "  - Restore backup if needed: git stash pop"
"""

        script_file = milestone_dir / "restore.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        script_file.chmod(0o755)
        
    def _update_milestone_index(self, milestone_name: str, description: str, commit_sha: str, timestamp: datetime):
        """Update the milestone index file."""
        milestones = []
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                milestones = json.load(f)
                
        milestone_record = {
            "name": milestone_name,
            "description": description,
            "commit_sha": commit_sha,
            "git_tag": f"milestone-{milestone_name}",
            "created": timestamp.isoformat(),
            "restoration_verified": False
        }
        
        # Remove existing milestone with same name
        milestones = [m for m in milestones if m['name'] != milestone_name]
        milestones.append(milestone_record)
        
        with open(self.index_file, 'w') as f:
            json.dump(milestones, f, indent=2)
            
    def _backup_current_state(self, backup_name: str):
        """Backup current state before restoration."""
        try:
            subprocess.run(["git", "stash", "push", "-m", backup_name], cwd=self.workspace_root)
            print(f"💾 Current state backed up as: {backup_name}")
        except subprocess.CalledProcessError:
            print("⚠️ Could not backup current state (no changes to stash)")
            
    def _verify_restoration(self, milestone_dir: Path) -> bool:
        """Verify that restoration completed successfully."""
        try:
            # Check if critical files exist
            critical_paths = [
                "sage-forge-professional/src/sage_forge",
                "sage-forge-professional/cli",
                "sage-forge-professional/README.md"
            ]
            
            for path in critical_paths:
                if not (self.workspace_root / path).exists():
                    print(f"⚠️ Missing critical path: {path}")
                    return False
                    
            # Try to run structure validation if available
            test_file = self.workspace_root / "sage-forge-professional/tests/test_professional_structure.py"
            if test_file.exists():
                result = subprocess.run(
                    ["python", str(test_file)], 
                    cwd=self.workspace_root / "sage-forge-professional",
                    capture_output=True
                )
                if result.returncode != 0:
                    print("⚠️ Structure validation failed")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="SAGE-Forge Milestone Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create milestone
    create_parser = subparsers.add_parser('create', help='Create new milestone')
    create_parser.add_argument('description', help='Milestone description')
    create_parser.add_argument('--force', action='store_true', help='Overwrite existing milestone')
    
    # List milestones
    list_parser = subparsers.add_parser('list', help='List all milestones')
    
    # Restore milestone
    restore_parser = subparsers.add_parser('restore', help='Restore to milestone')
    restore_parser.add_argument('milestone_name', help='Name of milestone to restore')
    restore_parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    manager = MilestoneManager()
    
    if args.command == 'create':
        manager.create_milestone(args.description, args.force)
    elif args.command == 'list':
        manager.list_milestones()
    elif args.command == 'restore':
        manager.restore_milestone(args.milestone_name, args.yes)


if __name__ == "__main__":
    main()