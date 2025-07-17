#!/usr/bin/env python3
"""
🎨 Revolutionary Code Formatter - Bypass Claude Code hooks entirely!

This script provides instant, reliable code formatting without depending on
Claude Code's problematic hook system. Run manually or integrate with your
preferred workflow.
"""

import subprocess
import sys
from pathlib import Path


class RevolutionaryFormatter:
    """A bulletproof code formatter that actually works."""
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.python_files = []
        
    def find_python_files(self, directories: list[str] = None) -> list[Path]:
        """Find all Python files in specified directories."""
        if directories is None:
            directories = ["src", "examples", "strategies", "nautilus_test"]
            
        python_files = []
        for directory in directories:
            dir_path = self.project_root / directory
            if dir_path.exists():
                python_files.extend(dir_path.rglob("*.py"))
                
        return python_files
    
    def format_with_ruff(self, files: list[Path]) -> bool:
        """Format files with ruff - the fast, reliable way."""
        if not files:
            print("📝 No Python files found to format")
            return True
            
        file_paths = [str(f) for f in files]
        
        try:
            print(f"🎯 Formatting {len(files)} Python files with ruff...")
            
            # Run ruff check with fixes
            check_result = subprocess.run(
                ["uv", "run", "ruff", "check", "--fix"] + file_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Run ruff format
            format_result = subprocess.run(
                ["uv", "run", "ruff", "format"] + file_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Ruff returns 0 for success, 1 for linting issues found (but still formatted)
            check_ok = check_result.returncode <= 1  # 0 = clean, 1 = issues found but processed
            format_ok = format_result.returncode == 0
            
            if check_ok and format_ok:
                print("✅ All files processed successfully!")
                
                # Show check results
                if check_result.stdout.strip():
                    if "All checks passed" in check_result.stdout:
                        print("   ✨ Code is clean - no issues found")
                    else:
                        print("   🔍 Issues found and processed:")
                        print(f"   {check_result.stdout.strip()}")
                
                # Show format results  
                if format_result.stdout.strip():
                    print(f"   📝 {format_result.stdout.strip()}")
                
                return True
            else:
                print("❌ Formatting failed:")
                print(f"   Check: code={check_result.returncode}, out='{check_result.stdout}', err='{check_result.stderr}'")
                print(f"   Format: code={format_result.returncode}, out='{format_result.stdout}', err='{format_result.stderr}'")
                return False
                
        except Exception as e:
            print(f"💥 Formatting failed with error: {e}")
            return False
    
    def format_recent_files(self, minutes: int = 30) -> bool:
        """Format files modified in the last N minutes."""
        import time
        
        all_files = self.find_python_files()
        cutoff_time = time.time() - (minutes * 60)
        
        recent_files = [
            f for f in all_files 
            if f.stat().st_mtime > cutoff_time
        ]
        
        if recent_files:
            print(f"🕒 Found {len(recent_files)} files modified in last {minutes} minutes")
            for f in recent_files:
                print(f"   • {f.relative_to(self.project_root)}")
            return self.format_with_ruff(recent_files)
        else:
            print(f"📭 No files modified in the last {minutes} minutes")
            return True
    
    def format_all(self) -> bool:
        """Format all Python files in the project."""
        all_files = self.find_python_files()
        return self.format_with_ruff(all_files)
        
    def format_specific_files(self, file_patterns: list[str]) -> bool:
        """Format specific files by pattern."""
        files = []
        for pattern in file_patterns:
            if "*" in pattern:
                files.extend(self.project_root.glob(pattern))
            else:
                file_path = Path(pattern)
                if not file_path.is_absolute():
                    file_path = self.project_root / file_path
                if file_path.exists():
                    files.append(file_path)
                    
        python_files = [f for f in files if f.suffix == ".py"]
        return self.format_with_ruff(python_files)


def main():
    """Revolutionary formatting - no hooks, no problems!"""
    formatter = RevolutionaryFormatter()
    
    if len(sys.argv) == 1:
        # Default: format recent files
        print("🚀 Revolutionary Formatter - Formatting recent files...")
        success = formatter.format_recent_files(30)
    elif sys.argv[1] == "--all":
        print("🚀 Revolutionary Formatter - Formatting ALL files...")
        success = formatter.format_all()
    elif sys.argv[1] == "--recent":
        minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        print(f"🚀 Revolutionary Formatter - Formatting files from last {minutes} minutes...")
        success = formatter.format_recent_files(minutes)
    else:
        # Format specific files
        print("🚀 Revolutionary Formatter - Formatting specific files...")
        success = formatter.format_specific_files(sys.argv[1:])
    
    if success:
        print("🎉 Revolutionary formatting complete!")
        sys.exit(0)
    else:
        print("💥 Revolutionary formatting failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
