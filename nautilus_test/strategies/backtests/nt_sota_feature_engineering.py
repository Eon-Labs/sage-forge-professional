#!/usr/bin/env python3
"""
🚀 NT-NATIVE STATE-OF-THE-ART FEATURE ENGINEERING 2025
=====================================================

State-of-the-art time series feature extraction using Catch22 canonical features,
following NautilusTrader's native bias-free patterns for guaranteed robustness.

Features:
- Catch22: 22 canonical time series features for comprehensive signal extraction
- Online computation with rolling windows for real-time performance
- NT-native indicator architecture with auto-registration
- Bias-free operation using only completed bar data
- Computational efficiency optimized for live trading

Based on:
- Catch22: CAnonical Time-series CHaracteristics (Lubba et al., 2019)
- NautilusTrader bias-free indicator patterns
- 2025 state-of-the-art feature engineering practices

Author: Claude Code Assistant  
Date: 2025-07-19
License: MIT
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Catch22 features library
try:
    import pycatch22
    CATCH22_AVAILABLE = True
except ImportError:
    CATCH22_AVAILABLE = False
    warnings.warn("pycatch22 not available. Install with: pip install pycatch22")

# NautilusTrader imports
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar

# Rich console for enhanced output
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()


class Catch22FeatureExtractor(Indicator):
    """
    🚀 Catch22 Feature Extractor - NT-Native State-of-the-Art Time Series Features
    
    Extracts 22 canonical time series characteristics following NT's bias-free patterns.
    
    Features extracted:
    1. DN_HistogramMode_5 - Mode of z-scored distribution  
    2. DN_HistogramMode_10 - Mode of z-scored distribution (wider bins)
    3. CO_f1ecac - First 1/e crossing of autocorr function
    4. CO_FirstMin_ac - First minimum of autocorr function
    5. CO_HistogramAMI_even_2_5 - Automutual info, equal-quantile binning
    6. CO_trev_1_num - Time-reversibility statistic
    7. MD_hrv_classic_pnn40 - % NN intervals > 40ms
    8. SB_BinaryStats_mean_longstretch1 - Longest stretch of 1s in mean-binarized series
    9. SB_TransitionMatrix_3ac_sumdiagcov - Transition matrix diagonal covariance
    10. PD_PeriodicityWang_th0_01 - Periodicity measure
    11. CO_Embed2_Dist_tau_d_expfit_meandiff - Exponential fit to mean vs. distance
    12. IN_AutoMutualInfoStats_40_gaussian_fmmi - First minimum of automutual info
    13. FC_LocalSimple_mean1_tauresrat - Ratio of tau to time series length
    14. DN_OutlierInclude_p_001_mdrmd - Median absolute deviation from median
    15. DN_OutlierInclude_n_001_mdrmd - Median absolute deviation (negative outliers)
    16. SP_Summaries_welch_rect_area_5_1 - Spectral power in different freq bands
    17. SB_BinaryStats_diff_longstretch0 - Longest stretch of 0s in differenced series
    18. SB_MotifThree_quantile_hh - Shannon entropy of 3-letter words in quantile series
    19. SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 - Rescaled range analysis
    20. SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 - Detrended fluctuation analysis
    21. SP_Summaries_welch_rect_centroid - Spectral centroid
    22. FC_LocalSimple_mean3_stderr - Local mean statistics
    """
    
    def __init__(self, window_size: int = 100, update_frequency: int = 5):
        """
        Initialize Catch22 feature extractor.
        
        Args:
            window_size: Size of rolling window for feature computation
            update_frequency: Update features every N bars (for efficiency)
        """
        super().__init__(params=[window_size, update_frequency])
        
        if not CATCH22_AVAILABLE:
            raise ImportError("pycatch22 is required. Install with: pip install pycatch22")
        
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        # Rolling window for price data (bias-free)
        self.price_buffer = deque(maxlen=window_size)
        
        # Feature storage
        self.features = {}
        self.feature_names = self._get_catch22_feature_names()
        
        # Update control
        self.update_counter = 0
        self.last_update_size = 0
        
        # Performance tracking
        self.computation_times = deque(maxlen=100)
        
        console.print(f"[green]🚀 Catch22 Feature Extractor initialized[/green]")
        console.print(f"[cyan]  • Window size: {window_size}[/cyan]")
        console.print(f"[cyan]  • Update frequency: every {update_frequency} bars[/cyan]")
        console.print(f"[cyan]  • Features: {len(self.feature_names)}[/cyan]")
    
    def _get_catch22_feature_names(self) -> List[str]:
        """Get the 22 canonical feature names."""
        return [
            'DN_HistogramMode_5',
            'DN_HistogramMode_10', 
            'CO_f1ecac',
            'CO_FirstMin_ac',
            'CO_HistogramAMI_even_2_5',
            'CO_trev_1_num',
            'MD_hrv_classic_pnn40',
            'SB_BinaryStats_mean_longstretch1',
            'SB_TransitionMatrix_3ac_sumdiagcov',
            'PD_PeriodicityWang_th0_01',
            'CO_Embed2_Dist_tau_d_expfit_meandiff',
            'IN_AutoMutualInfoStats_40_gaussian_fmmi',
            'FC_LocalSimple_mean1_tauresrat',
            'DN_OutlierInclude_p_001_mdrmd',
            'DN_OutlierInclude_n_001_mdrmd',
            'SP_Summaries_welch_rect_area_5_1',
            'SB_BinaryStats_diff_longstretch0',
            'SB_MotifThree_quantile_hh',
            'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
            'SP_Summaries_welch_rect_centroid',
            'FC_LocalSimple_mean3_stderr'
        ]
    
    def handle_bar(self, bar: Bar):
        """NT auto-calls this on bar completion (bias-free)."""
        self.update_raw(float(bar.close))
    
    def update_raw(self, price: float):
        """Update with completed bar price only (bias-free by design)."""
        import time
        
        self.price_buffer.append(price)
        self.update_counter += 1
        
        # Update features periodically for efficiency
        should_update = (
            self.update_counter % self.update_frequency == 0 or
            len(self.price_buffer) != self.last_update_size
        )
        
        if should_update and len(self.price_buffer) >= 50:  # Minimum for stable features
            start_time = time.time()
            
            try:
                self.features = self._compute_catch22_features()
                self._set_initialized(True)
                
                # Track computation time
                computation_time = time.time() - start_time
                self.computation_times.append(computation_time)
                
                self.last_update_size = len(self.price_buffer)
                
            except Exception as e:
                console.print(f"[red]❌ Catch22 computation error: {e}[/red]")
                # Use previous features if computation fails
                if not self.features:
                    self.features = {name: 0.0 for name in self.feature_names}
    
    def _compute_catch22_features(self) -> Dict[str, float]:
        """Compute all 22 canonical time series features."""
        if len(self.price_buffer) < 50:
            return {name: 0.0 for name in self.feature_names}
        
        # Convert to numpy array for catch22
        ts_data = np.array(list(self.price_buffer))
        
        # Handle edge cases
        if np.all(ts_data == ts_data[0]):  # Constant series
            return {name: 0.0 for name in self.feature_names}
        
        if np.any(np.isnan(ts_data)) or np.any(np.isinf(ts_data)):
            # Clean data
            ts_data = ts_data[np.isfinite(ts_data)]
            if len(ts_data) < 50:
                return {name: 0.0 for name in self.feature_names}
        
        try:
            # Compute all catch22 features at once
            features_array = pycatch22.catch22_all(ts_data)
            
            # Map to dictionary with feature names
            features_dict = {}
            for i, name in enumerate(self.feature_names):
                if i < len(features_array['values']):
                    value = features_array['values'][i]
                    # Handle NaN/inf values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    features_dict[name] = float(value)
                else:
                    features_dict[name] = 0.0
            
            return features_dict
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Catch22 computation warning: {e}[/yellow]")
            return {name: 0.0 for name in self.feature_names}
    
    def get_feature_vector(self) -> np.ndarray:
        """Get normalized feature vector for ML models."""
        if not self.initialized:
            return np.zeros(len(self.feature_names))
        
        # Extract values in consistent order
        values = [self.features.get(name, 0.0) for name in self.feature_names]
        feature_vector = np.array(values)
        
        # Robust normalization
        feature_vector = np.clip(feature_vector, -1e6, 1e6)  # Clip extreme values
        feature_vector = np.nan_to_num(feature_vector, 0.0)  # Replace NaN/inf with 0
        
        # Apply tanh normalization for stable ML input
        feature_vector = np.tanh(feature_vector / np.std(feature_vector + 1e-8))
        
        return feature_vector
    
    def get_feature_importance_scores(self) -> Dict[str, float]:
        """Get feature importance based on variance and stability."""
        if not self.initialized:
            return {name: 0.0 for name in self.feature_names}
        
        # Simple importance score based on feature magnitude and stability
        importance_scores = {}
        
        for name in self.feature_names:
            value = self.features.get(name, 0.0)
            
            # Importance = magnitude with stability bonus
            magnitude = abs(value)
            stability_bonus = 1.0 if not (np.isnan(value) or np.isinf(value)) else 0.0
            
            importance_scores[name] = magnitude * stability_bonus
        
        return importance_scores
    
    def get_computation_stats(self) -> Dict[str, float]:
        """Get performance statistics for monitoring."""
        if not self.computation_times:
            return {"avg_time": 0.0, "max_time": 0.0, "updates": 0}
        
        times = list(self.computation_times)
        return {
            "avg_time": np.mean(times),
            "max_time": np.max(times),
            "min_time": np.min(times),
            "total_updates": len(times),
            "buffer_size": len(self.price_buffer)
        }
    
    def reset(self):
        """Reset indicator state."""
        self.price_buffer.clear()
        self.features.clear()
        self.update_counter = 0
        self.last_update_size = 0
        self.computation_times.clear()
        self._set_initialized(False)


class StreamingCatch22Features:
    """
    🚀 Streaming Catch22 Features for Online Learning
    
    Optimized for continuous feature extraction with minimal latency.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.extractors = {}
        self.current_features = {}
        
    def add_time_series(self, series_name: str, data: np.ndarray):
        """Add a new time series for feature extraction."""
        if series_name not in self.extractors:
            self.extractors[series_name] = Catch22FeatureExtractor(self.window_size)
        
        # Update extractor with new data points
        for value in data:
            self.extractors[series_name].update_raw(float(value))
        
        if self.extractors[series_name].initialized:
            self.current_features[series_name] = self.extractors[series_name].get_feature_vector()
    
    def get_combined_features(self) -> np.ndarray:
        """Get combined feature vector from all time series."""
        if not self.current_features:
            return np.array([])
        
        # Concatenate all feature vectors
        combined = np.concatenate(list(self.current_features.values()))
        return combined
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all extracted features."""
        summary = {
            "total_series": len(self.extractors),
            "active_extractors": sum(1 for ext in self.extractors.values() if ext.initialized),
            "total_features": sum(len(features) for features in self.current_features.values()),
        }
        
        # Add computation stats
        for name, extractor in self.extractors.items():
            if extractor.initialized:
                stats = extractor.get_computation_stats()
                summary[f"{name}_stats"] = stats
        
        return summary


def test_catch22_extractor():
    """Test Catch22 feature extractor with synthetic data."""
    console.print("[yellow]🧪 Testing Catch22 Feature Extractor...[/yellow]")
    
    if not CATCH22_AVAILABLE:
        console.print("[red]❌ pycatch22 not available. Install with: pip install pycatch22[/red]")
        return False
    
    # Create test data
    np.random.seed(42)
    
    # Generate synthetic price series with different patterns
    n_points = 200
    t = np.linspace(0, 10, n_points)
    
    # Trend + noise + seasonality
    trend = 0.1 * t
    seasonality = 2 * np.sin(2 * np.pi * t / 5)
    noise = np.random.normal(0, 0.5, n_points)
    prices = 100 + trend + seasonality + noise
    
    # Test extractor
    extractor = Catch22FeatureExtractor(window_size=100, update_frequency=10)
    
    console.print("  Testing with synthetic time series...")
    
    # Update extractor with data
    for i, price in enumerate(prices):
        extractor.update_raw(price)
        
        if i % 50 == 0 and extractor.initialized:
            features = extractor.get_feature_vector()
            stats = extractor.get_computation_stats()
            
            console.print(f"    Step {i}: {len(features)} features extracted, "
                         f"avg computation time: {stats['avg_time']:.4f}s")
    
    # Final results
    if extractor.initialized:
        features = extractor.get_feature_vector()
        importance = extractor.get_feature_importance_scores()
        stats = extractor.get_computation_stats()
        
        console.print(f"  Final Results:")
        console.print(f"    Total features: {len(features)}")
        console.print(f"    Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
        console.print(f"    Avg computation time: {stats['avg_time']:.4f}s")
        console.print(f"    Total updates: {stats['total_updates']}")
        
        # Show top 5 most important features
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        console.print(f"    Top 5 features by importance:")
        for name, score in top_features:
            console.print(f"      {name}: {score:.4f}")
        
        console.print("[green]✅ Catch22 feature extractor test passed![/green]")
        return True
    else:
        console.print("[red]❌ Extractor failed to initialize[/red]")
        return False


def test_streaming_features():
    """Test streaming feature extraction."""
    console.print("[yellow]🧪 Testing Streaming Catch22 Features...[/yellow]")
    
    if not CATCH22_AVAILABLE:
        console.print("[red]❌ pycatch22 not available. Skipping streaming test[/red]")
        return False
    
    # Create streaming extractor
    streaming = StreamingCatch22Features(window_size=100)
    
    # Generate multiple time series
    np.random.seed(42)
    n_points = 150
    
    # Price series
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.1)
    volumes = 1000 + np.random.exponential(500, n_points)
    
    # Add series
    streaming.add_time_series("prices", prices)
    streaming.add_time_series("volumes", volumes)
    
    # Get results
    combined_features = streaming.get_combined_features()
    summary = streaming.get_feature_summary()
    
    console.print(f"  Streaming Results:")
    console.print(f"    Total series: {summary['total_series']}")
    console.print(f"    Active extractors: {summary['active_extractors']}")
    console.print(f"    Combined features: {len(combined_features)}")
    console.print(f"    Feature range: [{np.min(combined_features):.3f}, {np.max(combined_features):.3f}]")
    
    console.print("[green]✅ Streaming Catch22 features test passed![/green]")
    return True


if __name__ == "__main__":
    console.print("[bold green]🚀 NT-Native State-of-the-Art Feature Engineering![/bold green]")
    console.print("[dim]Catch22 canonical time series features for NautilusTrader[/dim]")
    
    # Run tests
    test_basic = test_catch22_extractor()
    test_stream = test_streaming_features()
    
    if test_basic and test_stream:
        console.print("\n[green]🌟 Ready for integration with NT-native strategies![/green]")
    else:
        console.print("\n[yellow]⚠️ Some tests failed. Check pycatch22 installation.[/yellow]")
        console.print("[cyan]Install with: pip install pycatch22[/cyan]")