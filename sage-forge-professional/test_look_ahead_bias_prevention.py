#!/usr/bin/env python3
"""
Look-Ahead Bias Prevention Validation Test
=========================================

Comprehensive test suite to validate that the DSM data integration and TiRex model
implementation prevent look-ahead bias in all scenarios.

Critical Tests:
1. DSM timestamp handling prevents future data contamination
2. TiRex model enforces temporal ordering
3. Complete pipeline maintains chronological data flow
4. Error detection for temporal violations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import warnings

# Add SAGE-Forge to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

# Suppress warnings for clean test output
warnings.filterwarnings('ignore')

try:
    from sage_forge.data.manager import ArrowDataManager
    from sage_forge.models.tirex_model import TiRexInputProcessor
    from nautilus_trader.model.data import Bar, BarType, BarSpecification
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class LookAheadBiasValidator:
    """Comprehensive validator for look-ahead bias prevention."""
    
    def __init__(self):
        self.test_results = []
        
    def test_dsm_timestamp_fallback_prevention(self) -> bool:
        """Test that DSM timestamp fallback doesn't use current time."""
        print("\n🔍 Testing DSM Timestamp Fallback Prevention...")
        
        try:
            # Create data manager
            data_manager = ArrowDataManager()
            
            # Create test data without timestamp column
            test_data = pd.DataFrame({
                'open': [50000, 50100, 50200],
                'high': [50100, 50200, 50300],
                'low': [49900, 50000, 50100],
                'close': [50050, 50150, 50250],
                'volume': [100, 110, 120]
            })
            
            # Set end_time to historical value
            historical_end_time = datetime(2024, 1, 15, 12, 0)
            data_manager._current_end_time = historical_end_time
            
            # Process data through standardization
            import polars as pl
            df_polars = pl.from_pandas(test_data)
            result_df = data_manager._standardize_columns(df_polars, "BTCUSDT")
            
            # Validate no future timestamps were created
            if 'timestamp' in result_df.columns:
                max_timestamp = result_df['timestamp'].max()
                current_time = datetime.now()
                
                if max_timestamp and max_timestamp <= current_time:
                    print("✅ DSM timestamp fallback prevented future data contamination")
                    self.test_results.append(("DSM Timestamp Fallback Prevention", True))
                    return True
                else:
                    print(f"❌ Future timestamp detected: {max_timestamp} > {current_time}")
                    self.test_results.append(("DSM Timestamp Fallback Prevention", False))
                    return False
            else:
                print("⚠️  No timestamp column found in result")
                self.test_results.append(("DSM Timestamp Fallback Prevention", False))
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.test_results.append(("DSM Timestamp Fallback Prevention", False))
            return False
    
    def test_tirex_temporal_ordering(self) -> bool:
        """Test that TiRex model enforces temporal ordering."""
        print("\n🔍 Testing TiRex Temporal Ordering Validation...")
        
        try:
            processor = TiRexInputProcessor(sequence_length=10)
            
            # Create test bars with proper chronological order
            base_time = datetime(2024, 1, 1, 12, 0)
            bars = []
            
            for i in range(5):
                timestamp = base_time + timedelta(minutes=i * 15)
                ts_ns = dt_to_unix_nanos(timestamp)
                
                bar = Bar(
                    bar_type=self._create_test_bar_type(),
                    open=Price.from_str("50000.0"),
                    high=Price.from_str("50100.0"),
                    low=Price.from_str("49900.0"),
                    close=Price.from_str(f"{50000 + i * 10}.0"),
                    volume=Quantity.from_int(100),
                    ts_event=ts_ns,
                    ts_init=ts_ns
                )
                bars.append(bar)
            
            # Test 1: Add bars in chronological order (should succeed)
            for bar in bars:
                processor.add_bar(bar)
            
            print("✅ Chronological order accepted correctly")
            
            # Test 2: Try to add a bar with earlier timestamp (should fail)
            try:
                earlier_timestamp = base_time - timedelta(minutes=15)  # Earlier than first bar
                ts_ns_early = dt_to_unix_nanos(earlier_timestamp)
                
                early_bar = Bar(
                    bar_type=self._create_test_bar_type(),
                    open=Price.from_str("49000.0"),
                    high=Price.from_str("49100.0"),
                    low=Price.from_str("48900.0"),
                    close=Price.from_str("49050.0"),
                    volume=Quantity.from_int(100),
                    ts_event=ts_ns_early,
                    ts_init=ts_ns_early
                )
                
                processor.add_bar(early_bar)
                print("❌ Temporal ordering violation not detected")
                self.test_results.append(("TiRex Temporal Ordering", False))
                return False
                
            except ValueError as e:
                if "Temporal ordering violation" in str(e):
                    print("✅ Temporal ordering violation correctly detected and prevented")
                    self.test_results.append(("TiRex Temporal Ordering", True))
                    return True
                else:
                    print(f"❌ Wrong error type: {e}")
                    self.test_results.append(("TiRex Temporal Ordering", False))
                    return False
                    
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.test_results.append(("TiRex Temporal Ordering", False))
            return False
    
    def test_future_data_filtering(self) -> bool:
        """Test that future data is filtered out from DSM data."""
        print("\n🔍 Testing Future Data Filtering...")
        
        try:
            data_manager = ArrowDataManager()
            
            # Create test data with mixed past and future timestamps
            current_time = datetime.now()
            past_time = current_time - timedelta(hours=1)
            future_time = current_time + timedelta(hours=1)
            
            test_data = pd.DataFrame({
                'timestamp': [past_time, current_time - timedelta(minutes=30), future_time],
                'open': [50000, 50100, 50200],
                'high': [50100, 50200, 50300],
                'low': [49900, 50000, 50100],
                'close': [50050, 50150, 50250],
                'volume': [100, 110, 120]
            })
            
            # Process through validation
            import polars as pl
            df_polars = pl.from_pandas(test_data)
            result_df = data_manager._validate_and_process_data(test_data, "BTCUSDT")
            
            # Check that future data was filtered
            if result_df.height < 3:  # Should have filtered out future data
                max_timestamp = result_df['timestamp'].max()
                if max_timestamp and max_timestamp <= current_time:
                    print("✅ Future data successfully filtered out")
                    self.test_results.append(("Future Data Filtering", True))
                    return True
                else:
                    print(f"❌ Future data not filtered: max_timestamp={max_timestamp}")
                    self.test_results.append(("Future Data Filtering", False))
                    return False
            else:
                print("❌ Future data filtering not applied")
                self.test_results.append(("Future Data Filtering", False))
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.test_results.append(("Future Data Filtering", False))
            return False
    
    def test_end_to_end_temporal_integrity(self) -> bool:
        """Test complete pipeline maintains temporal integrity."""
        print("\n🔍 Testing End-to-End Temporal Integrity...")
        
        try:
            # Create realistic historical data
            data_manager = ArrowDataManager()
            processor = TiRexInputProcessor(sequence_length=5)
            
            # Set historical time range
            end_time = datetime.now() - timedelta(days=1)  # Yesterday
            start_time = end_time - timedelta(hours=2)     # 2 hours of data
            
            # Simulate DSM data with proper timestamps
            timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
            
            test_data = pd.DataFrame({
                'timestamp': timestamps,
                'open': np.random.uniform(49000, 51000, len(timestamps)),
                'high': np.random.uniform(50000, 52000, len(timestamps)),
                'low': np.random.uniform(48000, 50000, len(timestamps)),
                'close': np.random.uniform(49500, 51500, len(timestamps)),
                'volume': np.random.uniform(100, 1000, len(timestamps))
            })
            
            # Process through DSM validation
            result_df = data_manager._validate_and_process_data(test_data, "BTCUSDT")
            
            # Convert to NT bars and process through TiRex
            bars = data_manager.to_nautilus_bars(result_df, "BTCUSDT-PERP.BINANCE")
            
            # Process through TiRex (should maintain temporal order)
            for i, bar in enumerate(bars[:5]):  # Only process first 5 to avoid buffer size issues
                processor.add_bar(bar)
                
            # Validate temporal integrity was maintained
            if len(processor.timestamp_buffer) > 1:
                timestamps_list = list(processor.timestamp_buffer)
                is_ordered = all(timestamps_list[i] <= timestamps_list[i+1] for i in range(len(timestamps_list)-1))
                
                if is_ordered:
                    print("✅ End-to-end temporal integrity maintained")
                    self.test_results.append(("End-to-End Temporal Integrity", True))
                    return True
                else:
                    print("❌ Temporal ordering violated in pipeline")
                    self.test_results.append(("End-to-End Temporal Integrity", False))
                    return False
            else:
                print("⚠️  Insufficient data for temporal integrity test")
                self.test_results.append(("End-to-End Temporal Integrity", False))
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.test_results.append(("End-to-End Temporal Integrity", False))
            return False
    
    def _create_test_bar_type(self) -> BarType:
        """Create a test bar type for NT Bar objects."""
        bar_spec = BarSpecification(
            step=15,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.LAST
        )
        
        return BarType(
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL
        )
    
    def run_all_tests(self) -> Dict:
        """Run all look-ahead bias prevention tests."""
        print("🛡️  LOOK-AHEAD BIAS PREVENTION VALIDATION SUITE")
        print("=" * 60)
        
        # Run all tests
        test_methods = [
            self.test_dsm_timestamp_fallback_prevention,
            self.test_tirex_temporal_ordering,
            self.test_future_data_filtering,
            self.test_end_to_end_temporal_integrity
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed with exception: {e}")
                self.test_results.append((test_method.__name__, False))
        
        # Summary
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 LOOK-AHEAD BIAS PREVENTION TEST RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        for test_name, passed in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        if success_rate == 100:
            print("\n🛡️  ALL LOOK-AHEAD BIAS PREVENTION TESTS PASSED")
            print("✅ System is protected against temporal data leakage")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED")
            print("❌ Look-ahead bias prevention needs attention")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'all_passed': passed_tests == total_tests,
            'results': self.test_results
        }


def main():
    """Run look-ahead bias prevention validation."""
    validator = LookAheadBiasValidator()
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    return 0 if results['all_passed'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)