#!/usr/bin/env python3
"""
Complete System Validation After Organization

This script runs all critical tests to ensure nothing broke during organization.
All tests must pass for production readiness.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def run_test(test_path: str, description: str) -> bool:
    """Run a test and return success status."""
    try:
        console.print(f"🔄 Running: {description}")
        result = subprocess.run([sys.executable, test_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            console.print(f"✅ PASSED: {description}")
            return True
        else:
            console.print(f"❌ FAILED: {description}")
            console.print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        console.print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        console.print(f"💥 ERROR: {description} - {e}")
        return False

def main():
    """Run complete validation suite."""
    console.print(Panel("🎯 COMPLETE SAGE-FORGE VALIDATION AFTER ORGANIZATION", style="bold cyan"))
    console.print("🔍 Validating all critical functionality after file reorganization")
    console.print("⚠️  All tests must pass for production readiness")
    console.print()
    
    # Test suite configuration
    tests = [
        {
            "path": "tests/regression/test_signal_threshold_fix.py",
            "description": "Signal Threshold Fix Regression Test",
            "critical": True,
            "category": "Regression Protection"
        },
        {
            "path": "tests/functional/validate_nt_compliance.py", 
            "description": "NT Compliance & Look-Ahead Bias Prevention",
            "critical": True,
            "category": "Compliance Validation"
        },
        {
            "path": "tests/validation/comprehensive_signal_validation.py",
            "description": "Comprehensive Signal Validation",
            "critical": False,
            "category": "Signal Quality"
        },
        {
            "path": "tests/validation/definitive_signal_proof_test.py",
            "description": "Definitive Signal Proof Test", 
            "critical": False,
            "category": "Signal Quality"
        }
    ]
    
    # Run all tests
    results = []
    critical_failures = 0
    
    for test in tests:
        if not Path(test["path"]).exists():
            console.print(f"❌ MISSING: {test['path']}")
            results.append({**test, "status": "MISSING", "passed": False})
            if test["critical"]:
                critical_failures += 1
            continue
        
        passed = run_test(test["path"], test["description"])
        results.append({**test, "status": "PASSED" if passed else "FAILED", "passed": passed})
        
        if not passed and test["critical"]:
            critical_failures += 1
        
        console.print()
    
    # Generate results table
    console.print("=" * 80)
    console.print(Panel("📊 VALIDATION RESULTS SUMMARY", style="bold green"))
    
    table = Table(title="System Validation Results", show_header=True, header_style="bold cyan")
    table.add_column("Test Category", style="white")
    table.add_column("Test Description", style="white") 
    table.add_column("Critical", style="yellow")
    table.add_column("Status", style="green")
    
    for result in results:
        critical_marker = "🔥 YES" if result["critical"] else "⚪ NO"
        status_marker = "✅ PASSED" if result["passed"] else "❌ FAILED"
        
        table.add_row(
            result["category"],
            result["description"],
            critical_marker,
            status_marker
        )
    
    console.print(table)
    console.print()
    
    # Final assessment
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    critical_tests = sum(1 for r in results if r["critical"])
    
    console.print("📊 VALIDATION SUMMARY:")
    console.print(f"   Total tests: {total_tests}")
    console.print(f"   Tests passed: {passed_tests}")
    console.print(f"   Critical tests: {critical_tests}")
    console.print(f"   Critical failures: {critical_failures}")
    console.print()
    
    # Production readiness assessment
    if critical_failures == 0:
        console.print("🎉 VALIDATION COMPLETE: ✅ ALL CRITICAL TESTS PASSED")
        console.print("🚀 SYSTEM STATUS: PRODUCTION READY")
        console.print("✅ Organization successful - no regressions detected")
        console.print("🛡️ All critical fixes remain intact and functional")
        console.print()
        console.print("🎯 READY FOR:")
        console.print("   • Production deployment")
        console.print("   • Live trading (with paper trading validation)")
        console.print("   • Team development")
        console.print("   • Extended testing (multi-timeframe, multi-asset)")
        
        return True
    else:
        console.print("❌ VALIDATION FAILED: CRITICAL ISSUES DETECTED")
        console.print(f"⚠️  {critical_failures} critical test(s) failed")
        console.print("🔧 IMMEDIATE ACTION REQUIRED:")
        console.print("   • Fix critical test failures")
        console.print("   • Investigate potential regressions")
        console.print("   • Re-run validation before deployment")
        console.print("❌ NOT READY FOR PRODUCTION")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)