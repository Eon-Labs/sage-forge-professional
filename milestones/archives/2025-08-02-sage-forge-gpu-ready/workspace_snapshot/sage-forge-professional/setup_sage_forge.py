#!/usr/bin/env python3
"""
🔥 SAGE-Forge Self-Contained Setup Script

Self-contained installation and validation for SAGE-Forge trading system.
Ensures all components work correctly in any environment.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run command with error handling."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def validate_environment():
    """Validate environment setup."""
    print("🧪 Validating SAGE-Forge environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check critical files exist
    critical_files = [
        "demos/ultimate_complete_demo.py",
        "src/sage_forge/__init__.py",
        "pyproject.toml",
        "uv.lock"
    ]
    
    for file_path in critical_files:
        if not Path(file_path).exists():
            print(f"❌ Critical file missing: {file_path}")
            return False
        print(f"✅ Found: {file_path}")
    
    return True


def setup_dependencies():
    """Setup all dependencies using uv."""
    print("📦 Setting up dependencies...")
    
    commands = [
        ("uv sync", "Installing dependencies"),
        ("uv run python -c 'import nautilus_trader; print(f\"✅ NautilusTrader: {nautilus_trader.__version__}\")'", "Validating NautilusTrader"),
        ("uv run python -c 'import finplot; print(\"✅ FinPlot available\")'", "Validating FinPlot"),
        ("uv run python -c 'import rich; print(\"✅ Rich available\")'", "Validating Rich"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True


def validate_components():
    """Validate SAGE-Forge components."""
    print("🧪 Validating SAGE-Forge components...")
    
    validation_script = '''
import sys
sys.path.insert(0, "src")

try:
    from sage_forge import (
        ArrowDataManager,
        BinanceSpecificationManager, 
        EnhancedModernBarDataProvider,
        RealisticPositionSizer,
        FundingActor,
        FinplotActor,
        display_ultimate_performance_summary,
        get_config,
    )
    print("✅ All SAGE-Forge components imported successfully")
    print("✅ Self-contained setup validated")
except ImportError as e:
    print(f"❌ Component import failed: {e}")
    sys.exit(1)
'''
    
    return run_command(f"uv run python -c '{validation_script}'", "Component validation")


def main():
    """Main setup function."""
    print("🚀 SAGE-Forge Self-Contained Setup")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    # Setup dependencies
    if not setup_dependencies():
        print("❌ Dependency setup failed")
        sys.exit(1)
    
    # Validate components
    if not validate_components():
        print("❌ Component validation failed")
        sys.exit(1)
    
    print("\n🎉 SAGE-Forge Self-Contained Setup Complete!")
    print("=" * 50)
    print("🚀 Next steps:")
    print("  1. Run ultimate demo: uv run python demos/ultimate_complete_demo.py")
    print("  2. Run validation tests: uv run python tests/test_ultimate_demo.py")
    print("  3. Explore documentation: cat README.md")
    print("\n🔄 If anything breaks, restore working state:")
    print("  cd /Users/terryli/eon/nt && ./sage-forge-archive/RESTORE_WORKING_STATE.sh")


if __name__ == "__main__":
    main()