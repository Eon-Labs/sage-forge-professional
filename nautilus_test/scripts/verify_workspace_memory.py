#!/usr/bin/env python3
"""
Workspace Memory Verification Script

Verifies that the critical workspace paths committed to CLAUDE.md are correct
and accessible. This ensures Claude Code sessions will have accurate path memory.
"""

import sys
from pathlib import Path

def verify_critical_paths():
    """Verify all critical workspace paths exist and are correct."""
    
    print("🧠 VERIFYING WORKSPACE MEMORY PATHS")
    print("=" * 50)
    
    # Critical paths from CLAUDE.md memory
    workspace_root = Path("/Users/terryli/eon/nt/")
    project_dir = Path("/Users/terryli/eon/nt/nautilus_test/")
    
    paths_to_verify = [
        # Root paths
        (workspace_root, "WORKSPACE ROOT"),
        (project_dir, "PROJECT DIR"),
        
        # Key project files
        (workspace_root / "CLAUDE.md", "CLAUDE.md (memory file)"),
        (project_dir / "pyproject.toml", "Project config"),
        (project_dir / "src" / "nautilus_test", "Source package"),
        
        # Critical modules from memory
        (project_dir / "src" / "nautilus_test" / "utils" / "cache_config.py", "Cache config module"),
        (project_dir / "src" / "nautilus_test" / "utils" / "data_manager.py", "Data manager"),
        (project_dir / "src" / "nautilus_test" / "funding" / "provider.py", "Funding provider"),
        
        # Scripts and examples
        (project_dir / "examples" / "native_funding_complete.py", "Native funding example"),
        (project_dir / "examples" / "sandbox" / "enhanced_dsm_hybrid_integration.py", "Enhanced DSM script"),
        (project_dir / "scripts" / "test_cache_config.py", "Cache test script"),
        (project_dir / "scripts" / "migrate_cache.py", "Migration script"),
        
        # Documentation
        (project_dir / "docs" / "cache_management.md", "Cache documentation"),
    ]
    
    success = True
    
    for path, description in paths_to_verify:
        if path.exists():
            print(f"✅ {description}: {path}")
        else:
            print(f"❌ {description}: {path} (NOT FOUND)")
            success = False
    
    print("=" * 50)
    
    if success:
        print("🎉 ALL CRITICAL PATHS VERIFIED!")
        print("💾 Workspace memory is accurate and complete")
        print("🚀 Claude Code sessions will have correct path references")
    else:
        print("⚠️ SOME PATHS ARE MISSING!")
        print("🔧 Update CLAUDE.md memory if paths have changed")
    
    return success


def test_relative_vs_absolute():
    """Test that relative and absolute path assumptions work correctly."""
    
    print("\n🧪 TESTING PATH ASSUMPTIONS")
    print("=" * 50)
    
    # Get current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Test if we're in the expected workspace
    expected_cwd = Path("/Users/terryli/eon/nt")
    if cwd == expected_cwd:
        print("✅ Working directory matches CLAUDE.md memory")
    else:
        print(f"⚠️ Working directory differs from expected: {expected_cwd}")
    
    # Test relative path assumptions
    relative_tests = [
        ("examples/native_funding_complete.py", "Native funding example (relative)"),
        ("scripts/test_cache_config.py", "Cache test script (relative)"),
        ("nautilus_test/pyproject.toml", "Project config (relative)"),
    ]
    
    print("\nRelative path tests (from current directory):")
    for rel_path, description in relative_tests:
        full_path = cwd / rel_path
        if full_path.exists():
            print(f"✅ {description}: {rel_path}")
        else:
            print(f"❌ {description}: {rel_path} (would fail)")
    
    print("=" * 50)


def display_memory_summary():
    """Display the high-impact memory summary for quick reference."""
    
    print("\n📋 HIGH-IMPACT MEMORY SUMMARY")
    print("=" * 50)
    print("WORKSPACE ROOT: /Users/terryli/eon/nt/")
    print("PROJECT DIR: /Users/terryli/eon/nt/nautilus_test/")
    print("WORKING DIR: Always assume /Users/terryli/eon/nt/")
    print()
    print("✅ File operations: Use ABSOLUTE paths")
    print("✅ Bash operations: Use relative paths from workspace root")
    print("❌ Never use relative paths without context")
    print()
    print("Example correct commands:")
    print("  Read /Users/terryli/eon/nt/nautilus_test/src/nautilus_test/utils/data_manager.py")
    print("  Bash: uv run python examples/native_funding_complete.py")
    print("  Bash: uv run python scripts/test_cache_config.py")
    print("=" * 50)


if __name__ == "__main__":
    print("🧠 WORKSPACE MEMORY VERIFICATION")
    print("This script verifies that critical workspace paths are accessible")
    print("and that the memory committed to CLAUDE.md is accurate.\n")
    
    # Verify critical paths
    paths_ok = verify_critical_paths()
    
    # Test path assumptions
    test_relative_vs_absolute()
    
    # Display memory summary
    display_memory_summary()
    
    # Final status
    if paths_ok:
        print("\n🎯 VERIFICATION COMPLETE: Workspace memory is accurate!")
        sys.exit(0)
    else:
        print("\n⚠️ VERIFICATION FAILED: Some paths need attention!")
        sys.exit(1)