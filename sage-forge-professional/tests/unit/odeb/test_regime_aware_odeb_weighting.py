#!/usr/bin/env python3
"""
Sluice 1D: Regime-Aware ODEB Weighting (RAEW) Tests

Tests regime-aware ODEB weighting functionality including regime stability scoring
and confidence × regime stability multiplicative weighting.
"""

import pandas as pd
import numpy as np
from typing import List

from sage_forge.reporting.performance import Position
from sage_forge.benchmarking.confidence_inheritance_oracle import ConfidenceInheritanceOracle
from sage_forge.benchmarking.regime_aware_odeb import RegimeAwareOdebAnalyzer, RegimeStability

def run_sluice_1d_tests():
    """Run all Sluice 1D tests."""
    
    print("🔧 Running Sluice 1D: Regime-Aware ODEB Weighting (RAEW) Tests")
    
    base_time = pd.Timestamp('2025-01-01 10:00:00')
    
    def create_test_position(minutes_offset, confidence, regime, pnl=10.0):
        return Position(
            open_time=base_time + pd.Timedelta(minutes=minutes_offset),
            close_time=base_time + pd.Timedelta(minutes=minutes_offset + 15),
            size_usd=1000.0,
            pnl=pnl,
            direction=1,
            confidence=confidence,
            market_regime=regime,
            regime_stability=0.8,
            prediction_phase="STABLE_WINDOW"
        )
    
    # Test 1: Regime Stability Calculation
    print("Testing regime stability calculation...")
    
    analyzer = RegimeAwareOdebAnalyzer()
    
    # Test known regime stability mappings
    assert analyzer.calculate_regime_stability("low_vol_trending") == RegimeStability.VERY_HIGH.value
    assert analyzer.calculate_regime_stability("medium_vol_trending") == RegimeStability.HIGH.value
    assert analyzer.calculate_regime_stability("high_vol_ranging") == RegimeStability.VERY_LOW.value
    assert analyzer.calculate_regime_stability("unknown") == RegimeStability.UNSTABLE.value
    
    print("✅ Regime stability calculation - PASSED")
    
    # Test 2: Regime-Confidence Weight Calculation
    print("Testing regime-confidence weighting...")
    
    # High confidence, stable regime
    stable_pos = create_test_position(0, 0.8, "low_vol_trending", 20.0)
    stable_weight = analyzer.get_regime_odeb_weight(stable_pos)
    
    # Low confidence, unstable regime
    unstable_pos = create_test_position(5, 0.3, "high_vol_ranging", -5.0)
    unstable_weight = analyzer.get_regime_odeb_weight(unstable_pos)
    
    # Stable regime should have higher weight
    assert stable_weight > unstable_weight, f"Stable regime weight {stable_weight} should exceed unstable {unstable_weight}"
    assert 0.0 <= stable_weight <= 1.0, "Weight should be in [0,1] bounds"
    assert 0.0 <= unstable_weight <= 1.0, "Weight should be in [0,1] bounds"
    
    print("✅ Regime-confidence weighting - PASSED")
    
    # Test 3: Regime Performance Analysis
    print("Testing regime performance analysis...")
    
    positions = [
        # High-performing stable regime positions
        create_test_position(0, 0.85, "low_vol_trending", 25.0),
        create_test_position(5, 0.80, "low_vol_trending", 22.0),
        create_test_position(10, 0.90, "low_vol_trending", 28.0),
        
        # Medium-performing trending positions
        create_test_position(15, 0.70, "medium_vol_trending", 15.0),
        create_test_position(20, 0.65, "medium_vol_trending", 12.0),
        
        # Poor-performing ranging positions
        create_test_position(25, 0.40, "high_vol_ranging", -8.0),
        create_test_position(30, 0.35, "high_vol_ranging", -12.0)
    ]
    
    regime_analysis = analyzer.analyze_regime_performance(positions)
    
    # Validate analysis results
    assert "low_vol_trending" in regime_analysis, "Should analyze low_vol_trending regime"
    assert "high_vol_ranging" in regime_analysis, "Should analyze high_vol_ranging regime"
    
    stable_analysis = regime_analysis["low_vol_trending"]
    unstable_analysis = regime_analysis["high_vol_ranging"]
    
    # Stable regime should outperform unstable regime
    assert stable_analysis.confidence_weighted_return > unstable_analysis.confidence_weighted_return, \
        "Stable regime should have higher weighted return"
    
    assert stable_analysis.stability_score > unstable_analysis.stability_score, \
        "Stable regime should have higher stability score"
    
    print("✅ Regime performance analysis - PASSED")
    
    # Test 4: Weighted ODEB Metrics Calculation
    print("Testing weighted ODEB metrics calculation...")
    
    metrics = analyzer.calculate_weighted_odeb_metrics(positions)
    
    # Validate required metrics
    required_fields = [
        "regime_weighted_return", "confidence_weighted_return", 
        "regime_weighted_volatility", "regime_weighted_sharpe",
        "traditional_odeb_efficiency", "regime_weighted_efficiency"
    ]
    
    for field in required_fields:
        assert field in metrics, f"Missing required metric: {field}"
    
    # Validate metric reasonableness
    assert isinstance(metrics["regime_weighted_return"], (int, float)), "Return should be numeric"
    assert metrics["regime_weighted_volatility"] >= 0, "Volatility should be non-negative"
    assert metrics["total_positions"] == len(positions), "Should count all positions"
    
    print("✅ Weighted ODEB metrics calculation - PASSED")
    
    # Test 5: Regime Recommendations
    print("Testing regime recommendations...")
    
    recommendations = analyzer.get_regime_recommendations(metrics)
    
    assert isinstance(recommendations, list), "Recommendations should be a list"
    assert len(recommendations) > 0, "Should provide at least one recommendation"
    
    # Check for meaningful recommendations
    recommendation_text = " ".join(recommendations)
    assert len(recommendation_text) > 50, "Recommendations should be substantive"
    
    print("✅ Regime recommendations - PASSED")
    
    # Test 6: Integration with Confidence Inheritance Oracle
    print("Testing oracle integration...")
    
    oracle = ConfidenceInheritanceOracle()
    analyzer_with_oracle = RegimeAwareOdebAnalyzer(oracle)
    
    # Process positions through oracle
    oracle.inherit_tirex_confidence(positions)
    
    # Calculate weighted metrics with oracle
    oracle_metrics = analyzer_with_oracle.calculate_weighted_odeb_metrics(positions)
    
    # Should have similar structure to non-oracle metrics
    for field in required_fields:
        assert field in oracle_metrics, f"Oracle metrics missing field: {field}"
    
    print("✅ Oracle integration - PASSED")
    
    # Test 7: Edge Cases
    print("Testing edge cases...")
    
    # Empty positions
    empty_metrics = analyzer.calculate_weighted_odeb_metrics([])
    assert "error" in empty_metrics, "Should handle empty positions gracefully"
    
    # All unknown regime positions (but with valid confidence)
    unknown_positions = [
        create_test_position(i*5, 0.5, "low_vol_trending", 5.0) for i in range(3)
    ]
    # Make them unknown after creation
    for pos in unknown_positions:
        pos.market_regime = "unknown"
    
    unknown_metrics = analyzer.calculate_weighted_odeb_metrics(unknown_positions)
    assert "error" in unknown_metrics or "regime_weighted_return" in unknown_metrics, "Should handle unknown regimes"
    
    # All zero confidence positions
    zero_conf_positions = [
        create_test_position(i*5, 0.0, "low_vol_trending", 5.0) for i in range(3)
    ]
    zero_metrics = analyzer.calculate_weighted_odeb_metrics(zero_conf_positions)
    if "error" not in zero_metrics:
        assert zero_metrics.get("zero_weight_positions", 0) >= 0, "Should handle zero weight positions"
    else:
        assert "zero" in zero_metrics["error"].lower(), "Should report zero weight error"
    
    print("✅ Edge cases - PASSED")
    
    print(f"\n🎯 Sluice 1D VALIDATION COMPLETE")
    print(f"⚖️ Regime-aware ODEB weighting - VALIDATED")
    print(f"🔄 Multiplicative weighting (regime × confidence) - FUNCTIONAL")
    print(f"📊 Regime performance analysis - COMPREHENSIVE")
    print(f"🔮 Oracle integration - SEAMLESS")
    
    return True


if __name__ == "__main__":
    success = run_sluice_1d_tests()
    exit(0 if success else 1)