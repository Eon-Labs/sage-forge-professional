#!/usr/bin/env python3
"""
NautilusTrader Pattern Compliance Validation
==========================================

Comprehensive validation of NautilusTrader native pattern compliance across
all SAGE-Forge components based on the specifications in:
docs/implementation/backtesting/nt-patterns.md

Tests:
1. Strategy Pattern Compliance
2. Actor Pattern Compliance  
3. Integration Pattern Compliance
4. Configuration Pattern Compliance
5. Data Flow Pattern Compliance
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

# Add SAGE-Forge to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

try:
    # NT imports for pattern validation
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.common.actor import Actor
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.enums import OrderSide, PositionSide
    
    # SAGE-Forge imports
    from sage_forge.strategies.tirex_sage_strategy import TiRexSageStrategy
    from sage_forge.visualization.native_finplot_actor import NativeFinplotActor
    from sage_forge.funding.actor import FundingActor
    
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class NTPatternComplianceValidator:
    """Validates NautilusTrader pattern compliance."""
    
    def __init__(self):
        self.validation_results = []
        
    def validate_strategy_pattern_compliance(self) -> bool:
        """Validate Strategy pattern compliance per nt-patterns.md checklist."""
        print("\n🔍 Validating Strategy Pattern Compliance...")
        
        compliance_checks = []
        
        # Check 1: Inherits from nautilus_trader.trading.strategy.Strategy
        try:
            if issubclass(TiRexSageStrategy, Strategy):
                print("✅ Strategy inherits from nautilus_trader.trading.strategy.Strategy")
                compliance_checks.append(True)
            else:
                print("❌ Strategy does not inherit from Strategy base class")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking Strategy inheritance: {e}")
            compliance_checks.append(False)
        
        # Check 2: Implements required methods (on_start, on_bar, on_stop)
        required_methods = ['on_start', 'on_bar', 'on_stop']
        strategy_methods = [method for method in dir(TiRexSageStrategy) if not method.startswith('_')]
        
        for method in required_methods:
            if method in strategy_methods:
                print(f"✅ Strategy implements {method}()")
                compliance_checks.append(True)
            else:
                print(f"❌ Strategy missing required method: {method}()")
                compliance_checks.append(False)
        
        # Check 3: Uses proper subscription patterns
        try:
            # Check if on_start method contains subscribe_bars call
            source = inspect.getsource(TiRexSageStrategy.on_start)
            if 'subscribe_bars' in source:
                print("✅ Strategy uses proper subscription patterns")
                compliance_checks.append(True)
            else:
                print("❌ Strategy missing proper subscription patterns")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking subscription patterns: {e}")
            compliance_checks.append(False)
        
        # Check 4: Handles orders through self.order_factory (or equivalent)
        try:
            # Check if strategy uses NT order handling
            source = inspect.getsource(TiRexSageStrategy._open_position)
            if 'MarketOrder' in source and 'submit_order' in source:
                print("✅ Strategy handles orders through NT order management")
                compliance_checks.append(True)
            else:
                print("❌ Strategy not using NT order management patterns")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking order handling: {e}")
            compliance_checks.append(False)
        
        # Check 5: Uses SAGE-Forge risk management
        try:
            source = inspect.getsource(TiRexSageStrategy.__init__)
            if 'RealisticPositionSizer' in source:
                print("✅ Strategy uses SAGE-Forge risk management")
                compliance_checks.append(True)
            else:
                print("❌ Strategy missing SAGE-Forge risk management")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking risk management: {e}")
            compliance_checks.append(False)
        
        success = all(compliance_checks)
        self.validation_results.append(("Strategy Pattern Compliance", success, compliance_checks))
        
        if success:
            print("✅ Strategy Pattern Compliance: PASS")
        else:
            print(f"❌ Strategy Pattern Compliance: FAIL ({sum(compliance_checks)}/{len(compliance_checks)} checks passed)")
        
        return success
    
    def validate_actor_pattern_compliance(self) -> bool:
        """Validate Actor pattern compliance per nt-patterns.md checklist."""
        print("\n🔍 Validating Actor Pattern Compliance...")
        
        compliance_checks = []
        actors_to_test = [
            ("NativeFinplotActor", NativeFinplotActor),
            ("FundingActor", FundingActor)
        ]
        
        for actor_name, actor_class in actors_to_test:
            print(f"\n   Testing {actor_name}...")
            
            # Check 1: Inherits from nautilus_trader.common.actor.Actor
            try:
                if issubclass(actor_class, Actor):
                    print(f"   ✅ {actor_name} inherits from nautilus_trader.common.actor.Actor")
                    compliance_checks.append(True)
                else:
                    print(f"   ❌ {actor_name} does not inherit from Actor base class")
                    compliance_checks.append(False)
            except Exception as e:
                print(f"   ❌ Error checking {actor_name} inheritance: {e}")
                compliance_checks.append(False)
            
            # Check 2: Implements required methods (on_start, on_stop)
            required_methods = ['on_start', 'on_stop']
            actor_methods = [method for method in dir(actor_class) if not method.startswith('_')]
            
            for method in required_methods:
                if method in actor_methods:
                    print(f"   ✅ {actor_name} implements {method}()")
                    compliance_checks.append(True)
                else:
                    print(f"   ❌ {actor_name} missing required method: {method}()")
                    compliance_checks.append(False)
            
            # Check 3: Uses NT logging system
            try:
                source = inspect.getsource(actor_class)
                if 'self.log' in source:
                    print(f"   ✅ {actor_name} uses NT logging system")
                    compliance_checks.append(True)
                else:
                    print(f"   ❌ {actor_name} not using NT logging system")
                    compliance_checks.append(False)
            except Exception as e:
                print(f"   ❌ Error checking {actor_name} logging: {e}")
                compliance_checks.append(False)
        
        success = all(compliance_checks)
        self.validation_results.append(("Actor Pattern Compliance", success, compliance_checks))
        
        if success:
            print("✅ Actor Pattern Compliance: PASS")
        else:
            print(f"❌ Actor Pattern Compliance: FAIL ({sum(compliance_checks)}/{len(compliance_checks)} checks passed)")
        
        return success
    
    def validate_integration_pattern_compliance(self) -> bool:
        """Validate Integration pattern compliance per nt-patterns.md checklist."""
        print("\n🔍 Validating Integration Pattern Compliance...")
        
        compliance_checks = []
        
        # Check 1: All imports work correctly
        try:
            # Test critical imports
            import sage_forge.strategies.tirex_sage_strategy
            import sage_forge.models.tirex_model
            import sage_forge.data.manager
            import sage_forge.backtesting.tirex_backtest_engine
            
            print("✅ All critical imports work correctly")
            compliance_checks.append(True)
        except ImportError as e:
            print(f"❌ Import error detected: {e}")
            compliance_checks.append(False)
        
        # Check 2: Components use NT cache system
        try:
            # Check if strategy uses NT cache
            strategy_source = inspect.getsource(TiRexSageStrategy)
            if 'self.cache' in strategy_source:
                print("✅ Components use NT cache system")
                compliance_checks.append(True)
            else:
                print("❌ Components not using NT cache system")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking NT cache usage: {e}")
            compliance_checks.append(False)
        
        # Check 3: Data flows through NT cache system
        try:
            # Check for cache access patterns
            if 'cache.instrument' in strategy_source or 'cache.position' in strategy_source:
                print("✅ Data flows through NT cache system")
                compliance_checks.append(True)
            else:
                print("❌ Data not flowing through NT cache system")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking data flow: {e}")
            compliance_checks.append(False)
        
        # Check 4: Orders execute through NT order management
        try:
            if 'submit_order' in strategy_source:
                print("✅ Orders execute through NT order management")
                compliance_checks.append(True)
            else:
                print("❌ Orders not using NT order management")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking order management: {e}")
            compliance_checks.append(False)
        
        # Check 5: Performance tracking via NT reporting
        try:
            if 'self.log' in strategy_source or 'console.print' in strategy_source:
                print("✅ Performance tracking implemented")
                compliance_checks.append(True)
            else:
                print("❌ Performance tracking missing")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking performance tracking: {e}")
            compliance_checks.append(False)
        
        success = all(compliance_checks)
        self.validation_results.append(("Integration Pattern Compliance", success, compliance_checks))
        
        if success:
            print("✅ Integration Pattern Compliance: PASS")
        else:
            print(f"❌ Integration Pattern Compliance: FAIL ({sum(compliance_checks)}/{len(compliance_checks)} checks passed)")
        
        return success
    
    def validate_configuration_pattern_compliance(self) -> bool:
        """Validate Configuration pattern compliance."""
        print("\n🔍 Validating Configuration Pattern Compliance...")
        
        compliance_checks = []
        
        # Check 1: Config handling follows NT patterns
        try:
            strategy_init_source = inspect.getsource(TiRexSageStrategy.__init__)
            
            # CRITICAL ISSUE: Check for complex config handling (lines 61-95 violation)
            lines = strategy_init_source.split('\n')
            config_handling_lines = len([line for line in lines if 'hasattr(config' in line or 'getattr(config' in line])
            
            if config_handling_lines > 5:  # Too complex config handling
                print(f"❌ Configuration handling too complex ({config_handling_lines} complex lines)")
                print("   Violates NT native configuration patterns")
                compliance_checks.append(False)
            else:
                print("✅ Configuration handling follows NT patterns")
                compliance_checks.append(True)
                
        except Exception as e:
            print(f"❌ Error checking configuration patterns: {e}")
            compliance_checks.append(False)
        
        # Check 2: Proper parameter validation
        try:
            if 'min_confidence' in strategy_init_source and 'max_position_size' in strategy_init_source:
                print("✅ Configuration parameters properly defined")
                compliance_checks.append(True)
            else:
                print("❌ Configuration parameters missing or improperly defined")
                compliance_checks.append(False)
        except Exception as e:
            print(f"❌ Error checking parameter validation: {e}")
            compliance_checks.append(False)
        
        success = all(compliance_checks)
        self.validation_results.append(("Configuration Pattern Compliance", success, compliance_checks))
        
        if success:
            print("✅ Configuration Pattern Compliance: PASS")
        else:
            print(f"❌ Configuration Pattern Compliance: FAIL ({sum(compliance_checks)}/{len(compliance_checks)} checks passed)")
        
        return success
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive NT pattern compliance validation."""
        print("🧭 NAUTILUSTRADER PATTERN COMPLIANCE VALIDATION")
        print("=" * 60)
        
        # Run all validation tests
        validation_methods = [
            self.validate_strategy_pattern_compliance,
            self.validate_actor_pattern_compliance,
            self.validate_integration_pattern_compliance,
            self.validate_configuration_pattern_compliance
        ]
        
        for validation_method in validation_methods:
            try:
                validation_method()
            except Exception as e:
                print(f"❌ Validation {validation_method.__name__} failed with exception: {e}")
                self.validation_results.append((validation_method.__name__, False, []))
        
        # Calculate overall compliance
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for _, passed, _ in self.validation_results if passed)
        compliance_rate = (passed_validations / total_validations) * 100 if total_validations > 0 else 0
        
        # Detailed results
        print(f"\n📊 NT PATTERN COMPLIANCE RESULTS:")
        print(f"   Validations Passed: {passed_validations}/{total_validations} ({compliance_rate:.1f}%)")
        
        for validation_name, passed, checks in self.validation_results:
            status = "✅ COMPLIANT" if passed else "❌ NON-COMPLIANT"
            if checks:
                detail = f"({sum(checks)}/{len(checks)} checks passed)"
            else:
                detail = "(validation failed)"
            print(f"   {validation_name}: {status} {detail}")
        
        # Overall assessment
        if compliance_rate == 100:
            print("\n🏆 ALL NT PATTERN COMPLIANCE VALIDATIONS PASSED")
            print("✅ System fully compliant with NautilusTrader native patterns")
        else:
            print(f"\n⚠️  {total_validations - passed_validations} PATTERN VIOLATIONS DETECTED")
            print("❌ NT pattern compliance needs attention")
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'compliance_rate': compliance_rate,
            'all_compliant': passed_validations == total_validations,
            'results': self.validation_results
        }


def main():
    """Run NT pattern compliance validation."""
    validator = NTPatternComplianceValidator()
    results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    return 0 if results['all_compliant'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)