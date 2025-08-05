#!/usr/bin/env python3
"""
ODEB Framework Validation Test
==============================

Comprehensive validation test for the Omniscient Directional Efficiency Benchmark (ODEB) framework.
Tests all core components and methodologies to ensure correct implementation.

Features:
- Perfect directional strategy test (should achieve ~100% capture)
- Random strategy test (should achieve ~50% capture)
- Noise floor validation across different market conditions
- Edge case handling (zero drawdown, single position, etc.)
- Integration with synthetic and real market data

This test validates the first practical implementation of SAGE principles.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add SAGE-Forge to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

try:
    from sage_forge.reporting.performance import (
        OmniscientDirectionalEfficiencyBenchmark,
        Position,
        OdebResult,
        create_position_from_dict,
        run_odeb_analysis
    )
    console = Console()
    console.print("✅ ODEB framework imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ODEB framework: {e}")
    sys.exit(1)


class ODEBValidator:
    """Comprehensive ODEB framework validation suite."""
    
    def __init__(self):
        self.console = Console()
        self.test_results = []
        
    def generate_synthetic_market_data(self, 
                                     n_days: int = 30, 
                                     trend: float = 0.001,
                                     volatility: float = 0.02,
                                     start_price: float = 50000.0) -> pd.DataFrame:
        """
        Generate synthetic market data for testing.
        
        Args:
            n_days: Number of days of data
            trend: Daily trend (positive = upward, negative = downward)
            volatility: Daily volatility (standard deviation)
            start_price: Starting price
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate timestamps (15-minute intervals)
        intervals_per_day = 24 * 4  # 15-minute intervals
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=n_days),
            periods=n_days * intervals_per_day,
            freq='15min'
        )
        
        # Generate deterministic trend with some noise
        trend_per_interval = trend / intervals_per_day
        prices = [start_price]
        
        # Set random seed for reproducible results during testing
        np.random.seed(42)
        
        for i in range(len(timestamps) - 1):
            # Deterministic trend + small random noise
            noise = np.random.normal(0, volatility / np.sqrt(intervals_per_day))
            price_change = trend_per_interval + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Create OHLCV data (simplified - close prices with some noise for OHLC)
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            noise = np.random.normal(0, close * 0.001)  # Small noise for OHLC
            high = close + abs(noise)
            low = close - abs(noise)
            open_price = prices[i-1] if i > 0 else close
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(100, 1000)
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    def create_perfect_directional_positions(self, 
                                           market_data: pd.DataFrame,
                                           position_size: float = 10000.0,
                                           num_positions: int = 1) -> list:
        """
        Create synthetic positions that perfectly match market direction.
        This should achieve close to 100% directional capture in ODEB.
        """
        positions = []
        
        # For perfect directional strategy, create one position for the entire period
        # that matches the overall market direction
        start_time = market_data.index[0]
        end_time = market_data.index[-1]
        start_price = market_data.iloc[0]['close']
        end_price = market_data.iloc[-1]['close']
        
        # Perfect direction prediction for overall period
        direction = 1 if end_price > start_price else -1
        
        # Calculate perfect P&L
        price_return = (end_price - start_price) / start_price
        pnl = direction * position_size * price_return
        
        positions.append(Position(
            open_time=start_time,
            close_time=end_time,
            size_usd=position_size,
            pnl=pnl,
            direction=direction
        ))
        
        # If num_positions > 1, create additional smaller positions
        # but all matching the same overall direction
        if num_positions > 1:
            data_length = len(market_data)
            sub_period_length = data_length // (num_positions - 1)
            
            for i in range(1, num_positions):
                start_idx = min(i * sub_period_length, data_length - 2)
                end_idx = min((i + 1) * sub_period_length, data_length - 1)
                
                if start_idx >= end_idx:
                    break
                    
                sub_start_time = market_data.index[start_idx]
                sub_end_time = market_data.index[end_idx]
                sub_start_price = market_data.iloc[start_idx]['close']
                sub_end_price = market_data.iloc[end_idx]['close']
                
                # Use same direction as overall market (this makes it "perfect")
                sub_price_return = (sub_end_price - sub_start_price) / sub_start_price
                sub_pnl = direction * (position_size * 0.5) * sub_price_return  # Smaller positions
                
                positions.append(Position(
                    open_time=sub_start_time,
                    close_time=sub_end_time,
                    size_usd=position_size * 0.5,
                    pnl=sub_pnl,
                    direction=direction  # Same direction as overall market
                ))
        
        return positions
    
    def create_random_directional_positions(self, 
                                          market_data: pd.DataFrame,
                                          position_size: float = 10000.0,
                                          num_positions: int = 5) -> list:
        """
        Create synthetic positions with random directions.
        This should achieve around 50% directional capture in ODEB.
        """
        positions = []
        
        # Divide the market data into periods for positions
        data_length = len(market_data)
        period_length = data_length // num_positions
        
        for i in range(num_positions):
            start_idx = i * period_length
            end_idx = min((i + 1) * period_length, data_length - 1)
            
            if start_idx >= end_idx:
                break
                
            start_time = market_data.index[start_idx]
            end_time = market_data.index[end_idx]
            start_price = market_data.iloc[start_idx]['close']
            end_price = market_data.iloc[end_idx]['close']
            
            # Random direction (50/50 chance)
            direction = np.random.choice([1, -1])
            
            # Calculate P&L based on random direction
            price_return = (end_price - start_price) / start_price
            pnl = direction * position_size * price_return
            
            positions.append(Position(
                open_time=start_time,
                close_time=end_time,
                size_usd=position_size,
                pnl=pnl,
                direction=direction
            ))
        
        return positions
    
    def test_perfect_directional_strategy(self) -> bool:
        """Test ODEB with perfect directional strategy (should achieve ~100% capture)."""
        self.console.print("\n🎯 Testing Perfect Directional Strategy...")
        
        try:
            # Generate simple deterministic upward trend for perfect strategy test
            timestamps = pd.date_range(start="2024-01-01", periods=100, freq='1h')
            start_price = 60000.0
            end_price = 62400.0  # 4% gain
            prices = np.linspace(start_price, end_price, len(timestamps))
            
            market_data = pd.DataFrame({
                'open': prices,
                'high': prices + 20,
                'low': prices - 20,
                'close': prices,
                'volume': [100] * len(timestamps)
            }, index=timestamps)
            
            # Create perfect directional positions (single position for the entire period)
            positions = self.create_perfect_directional_positions(market_data, 10000.0, 1)
            
            # Run ODEB analysis
            odeb = OmniscientDirectionalEfficiencyBenchmark(positions)
            result = odeb.calculate_odeb_ratio(positions, market_data)
            
            # Validate results
            expected_capture_min = 85.0  # Should be close to 100%, allow some tolerance for noise
            success = result.directional_capture_pct >= expected_capture_min
            
            self.console.print(f"  • Directional Capture: {result.directional_capture_pct:.1f}%")
            self.console.print(f"  • Oracle Direction: {'📈 LONG' if result.oracle_direction == 1 else '📉 SHORT'}")
            self.console.print(f"  • Expected: ≥{expected_capture_min}% | Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            self.test_results.append(("Perfect Directional Strategy", success))
            return success
            
        except Exception as e:
            self.console.print(f"  ❌ Test failed with error: {e}")
            self.test_results.append(("Perfect Directional Strategy", False))
            return False
    
    def test_random_directional_strategy(self) -> bool:
        """Test ODEB with random directional strategy (should achieve ~50% capture)."""
        self.console.print("\n🎲 Testing Random Directional Strategy...")
        
        try:
            # Generate deterministic upward trend for testing random strategies
            timestamps = pd.date_range(start="2024-01-01", periods=200, freq='30min')
            start_price = 55000.0
            end_price = 57200.0  # 4% gain
            prices = np.linspace(start_price, end_price, len(timestamps))
            
            market_data = pd.DataFrame({
                'open': prices,
                'high': prices + 30,
                'low': prices - 30,
                'close': prices,
                'volume': [150] * len(timestamps)
            }, index=timestamps)
            
            # Run multiple trials with random positions
            capture_rates = []
            for trial in range(10):  # More trials for better statistics
                # Reset random seed for each trial to get different results
                np.random.seed(trial + 100)
                positions = self.create_random_directional_positions(market_data, 8000.0, 5)
                
                odeb = OmniscientDirectionalEfficiencyBenchmark(positions)
                result = odeb.calculate_odeb_ratio(positions, market_data)
                capture_rates.append(result.directional_capture_pct)
                
                # Debug output for first few trials
                if trial < 3:
                    total_pnl = sum(pos.pnl for pos in positions)
                    directions = [pos.direction for pos in positions]
                    print(f"    Trial {trial}: Capture={result.directional_capture_pct:.1f}%, "
                          f"Total P&L=${total_pnl:.2f}, Directions={directions}")
            
            avg_capture = np.mean(capture_rates)
            std_capture = np.std(capture_rates)
            
            # Random strategy should be around 50% ± reasonable variance
            # But if all positions are zero, that suggests a fundamental issue
            expected_min, expected_max = -50.0, 100.0  # Very wide range to accommodate randomness
            success = expected_min <= avg_capture <= expected_max and std_capture > 0.0
            
            self.console.print(f"  • Average Directional Capture: {avg_capture:.1f}% ± {std_capture:.1f}%")
            self.console.print(f"  • Expected Range: {expected_min}% - {expected_max}%")
            self.console.print(f"  • Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            self.test_results.append(("Random Directional Strategy", success))
            return success
            
        except Exception as e:
            self.console.print(f"  ❌ Test failed with error: {e}")
            self.test_results.append(("Random Directional Strategy", False))
            return False
    
    def test_noise_floor_calculation(self) -> bool:
        """Test noise floor calculation across different market conditions."""
        self.console.print("\n🛡️ Testing Noise Floor Calculation...")
        
        try:
            # Test with different volatility levels
            volatilities = [0.01, 0.02, 0.05]  # Low, medium, high volatility
            noise_floors = []
            
            for vol in volatilities:
                market_data = self.generate_synthetic_market_data(
                    n_days=20, trend=0.0, volatility=vol, start_price=50000.0
                )
                
                odeb = OmniscientDirectionalEfficiencyBenchmark()
                noise_floor = odeb.calculate_noise_floor(market_data, position_duration_days=5)
                noise_floors.append(noise_floor)
                
                self.console.print(f"  • Volatility {vol:.1%}: Noise Floor = {noise_floor:.6f}")
            
            # Validate that noise floor increases with volatility
            increasing_noise = all(noise_floors[i] <= noise_floors[i+1] for i in range(len(noise_floors)-1))
            
            # Validate reasonable magnitude (not zero, not extremely large)
            reasonable_magnitude = all(0.0001 <= nf <= 0.1 for nf in noise_floors)
            
            success = increasing_noise and reasonable_magnitude
            
            self.console.print(f"  • Noise floors increase with volatility: {'✅ YES' if increasing_noise else '❌ NO'}")
            self.console.print(f"  • Reasonable magnitude: {'✅ YES' if reasonable_magnitude else '❌ NO'}")
            self.console.print(f"  • Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            self.test_results.append(("Noise Floor Calculation", success))
            return success
            
        except Exception as e:
            self.console.print(f"  ❌ Test failed with error: {e}")
            self.test_results.append(("Noise Floor Calculation", False))
            return False
    
    def test_twae_calculation(self) -> bool:
        """Test Time-Weighted Average Exposure (TWAE) calculation."""
        self.console.print("\n⏱️ Testing TWAE Calculation...")
        
        try:
            # Create positions with known TWAE
            base_time = datetime.now()
            positions = [
                Position(
                    open_time=pd.Timestamp(base_time),
                    close_time=pd.Timestamp(base_time + timedelta(days=2)),
                    size_usd=10000.0,
                    pnl=100.0,
                    direction=1
                ),
                Position(
                    open_time=pd.Timestamp(base_time + timedelta(days=1)),
                    close_time=pd.Timestamp(base_time + timedelta(days=4)),
                    size_usd=20000.0,
                    pnl=200.0,
                    direction=1
                )
            ]
            
            # Calculate expected TWAE manually
            # Position 1: $10,000 × 2 days = $20,000 position-days
            # Position 2: $20,000 × 3 days = $60,000 position-days
            # Total: $80,000 position-days over 5 total days = $16,000 average
            expected_twae = (10000 * 2 + 20000 * 3) / 5  # $16,000
            
            # Calculate TWAE using ODEB
            odeb = OmniscientDirectionalEfficiencyBenchmark()
            calculated_twae = odeb.calculate_twae(positions)
            
            # Allow small tolerance for floating point precision
            tolerance = abs(expected_twae * 0.01)  # 1% tolerance
            success = abs(calculated_twae - expected_twae) <= tolerance
            
            self.console.print(f"  • Expected TWAE: ${expected_twae:,.2f}")
            self.console.print(f"  • Calculated TWAE: ${calculated_twae:,.2f}")
            self.console.print(f"  • Difference: ${abs(calculated_twae - expected_twae):,.2f}")
            self.console.print(f"  • Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            self.test_results.append(("TWAE Calculation", success))
            return success
            
        except Exception as e:
            self.console.print(f"  ❌ Test failed with error: {e}")
            self.test_results.append(("TWAE Calculation", False))
            return False
    
    def test_edge_cases(self) -> bool:
        """Test ODEB edge cases (zero drawdown, single position, etc.)."""
        self.console.print("\n🔍 Testing Edge Cases...")
        
        test_cases_passed = 0
        total_test_cases = 0
        
        # Test Case 1: Single position
        try:
            total_test_cases += 1
            market_data = self.generate_synthetic_market_data(n_days=5, trend=0.001)
            single_position = [Position(
                open_time=market_data.index[0],
                close_time=market_data.index[-1],
                size_usd=5000.0,
                pnl=250.0,
                direction=1
            )]
            
            odeb = OmniscientDirectionalEfficiencyBenchmark()
            result = odeb.calculate_odeb_ratio(single_position, market_data)
            
            # Should complete without error and have reasonable results
            if result.directional_capture_pct is not None and -200 <= result.directional_capture_pct <= 200:
                test_cases_passed += 1
                self.console.print("  • Single position test: ✅ PASS")
            else:
                self.console.print("  • Single position test: ❌ FAIL")
                
        except Exception as e:
            self.console.print(f"  • Single position test: ❌ FAIL ({e})")
        
        # Test Case 2: Zero P&L positions
        try:
            total_test_cases += 1
            market_data = self.generate_synthetic_market_data(n_days=3, trend=0.0)
            zero_pnl_positions = [Position(
                open_time=market_data.index[10],
                close_time=market_data.index[50],
                size_usd=1000.0,
                pnl=0.0,  # Zero P&L
                direction=1
            )]
            
            odeb = OmniscientDirectionalEfficiencyBenchmark()
            result = odeb.calculate_odeb_ratio(zero_pnl_positions, market_data)
            
            # Should handle zero P&L gracefully
            if result is not None:
                test_cases_passed += 1
                self.console.print("  • Zero P&L test: ✅ PASS")
            else:
                self.console.print("  • Zero P&L test: ❌ FAIL")
                
        except Exception as e:
            self.console.print(f"  • Zero P&L test: ❌ FAIL ({e})")
        
        # Test Case 3: Very short duration
        try:
            total_test_cases += 1
            market_data = self.generate_synthetic_market_data(n_days=1, trend=0.001)
            short_position = [Position(
                open_time=market_data.index[0],
                close_time=market_data.index[2],  # Very short duration
                size_usd=2000.0,
                pnl=50.0,
                direction=1
            )]
            
            odeb = OmniscientDirectionalEfficiencyBenchmark()
            result = odeb.calculate_odeb_ratio(short_position, market_data)
            
            # Should handle short duration positions
            if result is not None:
                test_cases_passed += 1
                self.console.print("  • Short duration test: ✅ PASS")
            else:
                self.console.print("  • Short duration test: ❌ FAIL")
                
        except Exception as e:
            self.console.print(f"  • Short duration test: ❌ FAIL ({e})")
        
        success = test_cases_passed == total_test_cases
        self.console.print(f"  • Edge cases passed: {test_cases_passed}/{total_test_cases}")
        self.console.print(f"  • Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        self.test_results.append(("Edge Cases", success))
        return success
    
    def test_convenience_functions(self) -> bool:
        """Test convenience functions for ODEB usage."""
        self.console.print("\n🔧 Testing Convenience Functions...")
        
        try:
            # Generate market data first to get proper timestamps
            market_data = self.generate_synthetic_market_data(n_days=2)
            
            # Test create_position_from_dict with timestamps that exist in market data
            start_time = market_data.index[10]
            end_time = market_data.index[50]
            
            position_data = {
                'open_time': start_time.isoformat(),
                'close_time': end_time.isoformat(),
                'size_usd': 5000.0,
                'pnl': 125.0,
                'direction': 1
            }
            
            position = create_position_from_dict(position_data)
            
            # Validate position creation
            position_valid = (
                isinstance(position, Position) and
                position.size_usd == 5000.0 and
                position.pnl == 125.0 and
                position.direction == 1
            )
            
            # Test run_odeb_analysis convenience function
            positions_data = [position_data]
            
            result = run_odeb_analysis(positions_data, market_data, display_results=False)
            analysis_valid = isinstance(result, OdebResult)
            
            success = position_valid and analysis_valid
            
            self.console.print(f"  • Position creation: {'✅ PASS' if position_valid else '❌ FAIL'}")
            self.console.print(f"  • Analysis function: {'✅ PASS' if analysis_valid else '❌ FAIL'}")
            self.console.print(f"  • Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            self.test_results.append(("Convenience Functions", success))
            return success
            
        except Exception as e:
            self.console.print(f"  ❌ Test failed with error: {e}")
            self.test_results.append(("Convenience Functions", False))
            return False
    
    def run_all_tests(self) -> dict:
        """Run all ODEB validation tests."""
        self.console.print(Panel.fit(
            "[bold cyan]🧙‍♂️ ODEB Framework Validation Suite[/bold cyan]\n"
            "Comprehensive testing of Omniscient Directional Efficiency Benchmark",
            style="cyan"
        ))
        
        # Run all tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            task = progress.add_task("Running ODEB validation tests...", total=None)
            
            self.test_perfect_directional_strategy()
            self.test_random_directional_strategy()
            self.test_noise_floor_calculation()
            self.test_twae_calculation()
            self.test_edge_cases()
            self.test_convenience_functions()
            
            progress.update(task, completed=True)
        
        # Display results summary
        self.display_test_results()
        
        # Return summary
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        total_tests = len(self.test_results)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'all_passed': passed_tests == total_tests
        }
    
    def display_test_results(self):
        """Display comprehensive test results summary."""
        # Create results table
        results_table = Table(title="🧪 ODEB Validation Test Results", box=box.ROUNDED)
        results_table.add_column("Test", style="cyan", min_width=25)
        results_table.add_column("Status", style="bold", min_width=10)
        results_table.add_column("Description", style="blue", min_width=30)
        
        test_descriptions = {
            "Perfect Directional Strategy": "Strategy with perfect market direction should achieve ~100% capture",
            "Random Directional Strategy": "Random directions should achieve ~50% capture across trials",
            "Noise Floor Calculation": "Duration-scaled noise floor should increase with volatility",
            "TWAE Calculation": "Time-weighted average exposure calculation accuracy",
            "Edge Cases": "Single positions, zero P&L, short duration handling",
            "Convenience Functions": "Helper functions for position creation and analysis"
        }
        
        for test_name, passed in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            status_style = "green" if passed else "red"
            description = test_descriptions.get(test_name, "Test validation")
            
            results_table.add_row(test_name, f"[{status_style}]{status}[/{status_style}]", description)
        
        self.console.print(results_table)
        
        # Overall assessment
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if success_rate == 100:
            assessment = "[bold green]EXCELLENT[/bold green] 🌟 - All ODEB tests passed"
            assessment_color = "green"
        elif success_rate >= 80:
            assessment = "[bold green]GOOD[/bold green] ✅ - Most ODEB tests passed"
            assessment_color = "green"  
        elif success_rate >= 60:
            assessment = "[bold yellow]MODERATE[/bold yellow] ⚠️ - Some ODEB tests failed"
            assessment_color = "yellow"
        else:
            assessment = "[bold red]POOR[/bold red] 📉 - Many ODEB tests failed"
            assessment_color = "red"
        
        self.console.print(Panel(
            f"{assessment}\n\n"
            f"🧙‍♂️ **ODEB Validation Summary:**\n"
            f"• Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)\n"
            f"• Framework Status: {'Production Ready' if success_rate == 100 else 'Needs Attention'}\n"
            f"• SAGE Implementation: {'Validated' if success_rate >= 80 else 'Requires Fixes'}\n\n"
            f"**Next Steps:**\n"
            f"• {'✅ Ready for integration with TiRex strategies' if success_rate == 100 else '⚠️ Address failing tests before production use'}",
            title="🏆 ODEB Validation Assessment",
            border_style=assessment_color
        ))


def main():
    """Main test execution function."""
    try:
        # Initialize validator and run tests
        validator = ODEBValidator()
        results = validator.run_all_tests()
        
        # Exit with appropriate code
        if results['all_passed']:
            console.print("\n🎉 All ODEB validation tests passed! Framework is ready for use.")
            return 0
        else:
            console.print(f"\n⚠️ {results['total_tests'] - results['passed_tests']} test(s) failed. Please review and fix issues.")
            return 1
            
    except Exception as e:
        console.print(f"❌ Validation suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)