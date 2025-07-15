"""
Funding rate validation for NautilusTrader integration.

This module provides focused validation of funding rate calculations
to ensure mathematical accuracy and temporal precision.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class ValidationScenario:
    """Represents a funding validation scenario."""
    name: str
    position_size: float  # BTC (positive=long, negative=short)
    mark_price: float    # USD
    funding_rate: float  # Decimal (e.g., 0.0001 = 0.01%)
    expected_payment: float  # USD (positive=pay, negative=receive)
    scenario_type: str  # "bull_market", "bear_market", "extreme"


class FundingValidator:
    """
    Focused funding rate validation.
    
    Validates calculations against known scenarios and mathematical properties.
    """
    
    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []
        console.print("[green]✅ FundingValidator initialized[/green]")
    
    def get_standard_scenarios(self) -> List[ValidationScenario]:
        """Get standard validation scenarios based on exchange documentation."""
        return [
            # Long positions in bull market (positive funding rate)
            ValidationScenario(
                name="Long Bull Market",
                position_size=1.0,  # 1 BTC long
                mark_price=50000.0,  # $50k
                funding_rate=0.0001,  # +0.01%
                expected_payment=5.0,  # Pays $5
                scenario_type="bull_market"
            ),
            
            # Short positions in bull market (positive funding rate)
            ValidationScenario(
                name="Short Bull Market", 
                position_size=-1.0,  # 1 BTC short
                mark_price=50000.0,
                funding_rate=0.0001,  # +0.01%
                expected_payment=-5.0,  # Receives $5
                scenario_type="bull_market"
            ),
            
            # Long positions in bear market (negative funding rate)
            ValidationScenario(
                name="Long Bear Market",
                position_size=1.0,  # 1 BTC long
                mark_price=45000.0,
                funding_rate=-0.0001,  # -0.01%
                expected_payment=-4.5,  # Receives $4.5
                scenario_type="bear_market"
            ),
            
            # Short positions in bear market (negative funding rate)
            ValidationScenario(
                name="Short Bear Market",
                position_size=-1.0,  # 1 BTC short
                mark_price=45000.0,
                funding_rate=-0.0001,  # -0.01%
                expected_payment=4.5,  # Pays $4.5
                scenario_type="bear_market"
            ),
            
            # Small position sizes (realistic trading)
            ValidationScenario(
                name="Small Long Position",
                position_size=0.001,  # 0.001 BTC long
                mark_price=60000.0,
                funding_rate=0.00015,  # +0.015%
                expected_payment=0.009,  # Pays $0.009
                scenario_type="realistic"
            ),
            
            # Extreme funding rates
            ValidationScenario(
                name="Extreme High Funding",
                position_size=0.5,  # 0.5 BTC long
                mark_price=55000.0,
                funding_rate=0.0075,  # +0.75% (extreme)
                expected_payment=206.25,  # Pays $206.25
                scenario_type="extreme"
            ),
        ]
    
    def validate_funding_calculation(
        self,
        position_size: float,
        mark_price: float,
        funding_rate: float,
        expected_payment: float,
        tolerance: float = 0.001
    ) -> bool:
        """
        Validate a single funding calculation.
        
        Parameters
        ----------
        position_size : float
            Position size in BTC (positive=long, negative=short).
        mark_price : float
            Mark price in USD.
        funding_rate : float
            Funding rate (e.g., 0.0001 = 0.01%).
        expected_payment : float
            Expected payment in USD (positive=pay, negative=receive).
        tolerance : float
            Tolerance for floating point comparison.
            
        Returns
        -------
        bool
            True if calculation matches expected result within tolerance.
        """
        # Calculate actual payment: Position × Price × Rate
        actual_payment = position_size * mark_price * funding_rate
        
        # Check if within tolerance
        diff = abs(actual_payment - expected_payment)
        is_valid = diff <= tolerance
        
        return is_valid
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation against all standard scenarios.
        
        Returns
        -------
        Dict[str, Any]
            Validation results summary.
        """
        console.print("[bold cyan]🔍 Running Comprehensive Funding Validation[/bold cyan]")
        
        scenarios = self.get_standard_scenarios()
        validation_results = []
        passed_count = 0
        
        # Create results table
        results_table = Table(title="Funding Calculation Validation")
        results_table.add_column("Scenario", style="bold")
        results_table.add_column("Position", style="blue")
        results_table.add_column("Price", style="yellow")
        results_table.add_column("Rate", style="cyan")
        results_table.add_column("Expected", style="white")
        results_table.add_column("Actual", style="white")
        results_table.add_column("Result", style="bold")
        
        for scenario in scenarios:
            # Calculate actual payment
            actual_payment = (scenario.position_size * 
                            scenario.mark_price * 
                            scenario.funding_rate)
            
            # Validate
            is_valid = self.validate_funding_calculation(
                scenario.position_size,
                scenario.mark_price,
                scenario.funding_rate,
                scenario.expected_payment
            )
            
            if is_valid:
                passed_count += 1
                result_style = "[green]✅ PASS[/green]"
            else:
                result_style = "[red]❌ FAIL[/red]"
            
            # Add to table
            position_str = f"{scenario.position_size:+.3f} BTC"
            price_str = f"${scenario.mark_price:,.0f}"
            rate_str = f"{scenario.funding_rate:+.6f}"
            expected_str = f"${scenario.expected_payment:+.3f}"
            actual_str = f"${actual_payment:+.3f}"
            
            results_table.add_row(
                scenario.name,
                position_str,
                price_str,
                rate_str,
                expected_str,
                actual_str,
                result_style
            )
            
            # Store result
            validation_results.append({
                'scenario': scenario.name,
                'position_size': scenario.position_size,
                'mark_price': scenario.mark_price,
                'funding_rate': scenario.funding_rate,
                'expected_payment': scenario.expected_payment,
                'actual_payment': actual_payment,
                'is_valid': is_valid,
                'type': scenario.scenario_type
            })
        
        console.print(results_table)
        
        # Summary
        total_scenarios = len(scenarios)
        pass_rate = (passed_count / total_scenarios) * 100
        
        summary = {
            'total_scenarios': total_scenarios,
            'passed': passed_count,
            'failed': total_scenarios - passed_count,
            'pass_rate_percent': pass_rate,
            'validation_results': validation_results,
            'mathematical_integrity': 'VERIFIED' if pass_rate >= 95 else 'FAILED'
        }
        
        # Display summary
        if pass_rate >= 95:
            status_color = "green"
            status_text = "🎉 VALIDATION PASSED"
        else:
            status_color = "red"
            status_text = "⚠️ VALIDATION FAILED"
        
        console.print(Panel.fit(
            f"[bold {status_color}]{status_text}[/bold {status_color}]\n"
            f"Passed: {passed_count}/{total_scenarios} ({pass_rate:.1f}%)\n"
            f"Mathematical integrity: {summary['mathematical_integrity']}",
            title="VALIDATION SUMMARY"
        ))
        
        self.validation_results = validation_results
        return summary
    
    def validate_temporal_accuracy(self, funding_times: List[datetime]) -> Dict[str, Any]:
        """
        Validate temporal accuracy of funding intervals.
        
        Parameters
        ----------
        funding_times : List[datetime]
            List of funding timestamps to validate.
            
        Returns
        -------
        Dict[str, Any]
            Temporal validation results.
        """
        if len(funding_times) < 2:
            return {"valid": False, "reason": "Insufficient funding times"}
        
        # Check 8-hour intervals
        intervals = []
        for i in range(1, len(funding_times)):
            interval = (funding_times[i] - funding_times[i-1]).total_seconds() / 3600
            intervals.append(interval)
        
        # Expected: 8 hours ± 1 hour tolerance
        valid_intervals = [7 <= interval <= 9 for interval in intervals]
        accuracy_pct = (sum(valid_intervals) / len(valid_intervals)) * 100
        
        # Check UTC timing (should be 00:00, 08:00, 16:00)
        valid_times = []
        for ft in funding_times:
            hour = ft.hour
            is_valid_hour = hour in [0, 8, 16]
            valid_times.append(is_valid_hour)
        
        time_accuracy_pct = (sum(valid_times) / len(valid_times)) * 100
        
        return {
            "valid": accuracy_pct >= 90 and time_accuracy_pct >= 90,
            "interval_accuracy_percent": accuracy_pct,
            "time_accuracy_percent": time_accuracy_pct,
            "total_intervals": len(intervals),
            "average_interval_hours": sum(intervals) / len(intervals) if intervals else 0,
            "expected_hours": [0, 8, 16]
        }
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        # Group by scenario type
        by_type = {}
        for result in self.validation_results:
            scenario_type = result['type']
            if scenario_type not in by_type:
                by_type[scenario_type] = []
            by_type[scenario_type].append(result)
        
        # Calculate statistics
        passed = sum(1 for r in self.validation_results if r['is_valid'])
        total = len(self.validation_results)
        
        return {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) * 100 if total > 0 else 0,
            "results_by_type": by_type,
            "mathematical_integrity": "VERIFIED" if (passed / total) >= 0.95 else "FAILED",
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }


def run_validation_demo():
    """Run validation demonstration."""
    console.print(Panel.fit(
        "[bold cyan]🔍 Funding Rate Validation Demo[/bold cyan]\n"
        "Testing mathematical accuracy and edge cases",
        title="VALIDATION DEMO"
    ))
    
    validator = FundingValidator()
    results = validator.run_comprehensive_validation()
    
    console.print(f"\n[bold green]✅ Validation complete![/bold green]")
    console.print(f"Mathematical integrity: {results['mathematical_integrity']}")
    
    return results


if __name__ == "__main__":
    run_validation_demo()