#!/usr/bin/env python3
"""
🧪 Professional Structure Validation Test

Validates that the complete professional hierarchy is properly implemented
with all components organized correctly.
"""

import sys
from pathlib import Path


def test_directory_structure():
    """Test that professional directory structure exists."""
    print("🧪 Testing professional directory structure...")
    
    required_dirs = [
        "demos",
        "src/sage_forge/core",
        "src/sage_forge/actors", 
        "src/sage_forge/strategies",
        "src/sage_forge/indicators",
        "src/sage_forge/data",
        "src/sage_forge/models",
        "src/sage_forge/risk",
        "src/sage_forge/visualization",
        "src/sage_forge/market",
        "src/sage_forge/reporting",
        "tests/integration",
        "tests/unit",
        "tests/functional",
        "cli",
        "configs",
        "documentation/api",
        "documentation/tutorials",
        "documentation/examples"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required directories exist")
    return True


def test_cli_tools():
    """Test that CLI tools are properly implemented.""" 
    print("🧪 Testing CLI tools...")
    
    required_cli = [
        "cli/sage-create",
        "cli/sage-validate"
    ]
    
    missing_cli = []
    for cli_path in required_cli:
        cli_file = Path(cli_path)
        if not cli_file.exists():
            missing_cli.append(cli_path)
        elif not cli_file.is_file() or not (cli_file.stat().st_mode & 0o111):
            missing_cli.append(f"{cli_path} (not executable)")
    
    if missing_cli:
        print(f"❌ Missing/broken CLI tools: {missing_cli}")
        return False
    
    print("✅ All CLI tools exist and are executable")
    return True


def test_configuration_files():
    """Test that configuration files exist."""
    print("🧪 Testing configuration files...")
    
    required_configs = [
        "configs/default_config.yaml",
        "configs/production_config.yaml"
    ]
    
    missing_configs = []
    for config_path in required_configs:
        if not Path(config_path).exists():
            missing_configs.append(config_path)
    
    if missing_configs:
        print(f"❌ Missing configuration files: {missing_configs}")
        return False
    
    print("✅ All configuration files exist")
    return True


def test_documentation():
    """Test that documentation exists."""
    print("🧪 Testing documentation...")
    
    required_docs = [
        "README.md",
        "documentation/NT_PATTERNS.md"
    ]
    
    missing_docs = []
    for doc_path in required_docs:
        if not Path(doc_path).exists():
            missing_docs.append(doc_path)
    
    if missing_docs:
        print(f"❌ Missing documentation: {missing_docs}")
        return False
    
    print("✅ All documentation exists")
    return True


def test_ultimate_demo_preserved():
    """Test that ultimate demo is preserved."""
    print("🧪 Testing ultimate demo preservation...")
    
    demo_path = Path("demos/ultimate_complete_demo.py")
    if not demo_path.exists():
        print("❌ Ultimate demo not found")
        return False
    
    # Check file size (should be substantial)
    file_size = demo_path.stat().st_size
    if file_size < 30000:
        print(f"❌ Ultimate demo too small: {file_size} bytes")
        return False
    
    print(f"✅ Ultimate demo preserved ({file_size:,} bytes)")
    return True


def test_no_duplicate_structure():
    """Test that there are no duplicate CLI directories."""
    print("🧪 Testing for duplicate structures...")
    
    # Check that CLI is not duplicated in src/
    duplicate_cli = Path("src/sage_forge/cli")
    if duplicate_cli.exists():
        print("❌ Duplicate CLI found in src/sage_forge/cli")
        return False
    
    print("✅ No duplicate structures found")
    return True


def main():
    """Run all professional structure tests."""
    print("🚀 SAGE-Forge Professional Structure Validation")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("CLI Tools", test_cli_tools),
        ("Configuration Files", test_configuration_files),
        ("Documentation", test_documentation),
        ("Ultimate Demo Preservation", test_ultimate_demo_preserved),
        ("No Duplicate Structures", test_no_duplicate_structure)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - Professional structure is complete!")
        print("🚀 Ready for professional development workflow")
    else:
        print("❌ Some tests failed - structure needs fixes")
        sys.exit(1)


if __name__ == "__main__":
    main()