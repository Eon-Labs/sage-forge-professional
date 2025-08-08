#!/usr/bin/env python3
"""
Sluice 1A: Position Dataclass Enhancement Unit Tests

Tests the enhanced Position dataclass with confidence and regime metadata fields.
Validates backward compatibility and new ODEB-specific functionality.
"""

import pandas as pd
import numpy as np

from sage_forge.reporting.performance import Position


def test_enhanced_position_creation_with_new_fields():
    """Test creating Position with enhanced ODEB fields."""
    
    position = Position(
        open_time=pd.Timestamp('2025-01-01 10:00:00'),
        close_time=pd.Timestamp('2025-01-01 10:15:00'),
        size_usd=1000.0,
        pnl=25.5,
        direction=1,
        confidence=0.75,
        market_regime="low_vol_trending", 
        regime_stability=0.85,
        prediction_phase="STABLE_WINDOW"
    )
    
    # Validate core fields
    assert position.size_usd == 1000.0
    assert position.pnl == 25.5
    assert position.direction == 1
    
    # Validate enhanced fields
    assert position.confidence == 0.75
    assert position.market_regime == "low_vol_trending"
    assert position.regime_stability == 0.85
    assert position.prediction_phase == "STABLE_WINDOW"
    
    print("✅ Enhanced position creation - PASSED")


def test_position_backward_compatibility():
    """Test Position creation with only core fields (backward compatibility)."""
    
    # Create position with only original fields
    position = Position(
        open_time=pd.Timestamp('2025-01-01 11:00:00'),
        close_time=pd.Timestamp('2025-01-01 11:30:00'),
        size_usd=500.0,
        pnl=-12.3,
        direction=-1
    )
    
    # Core fields should work
    assert position.size_usd == 500.0
    assert position.pnl == -12.3
    assert position.direction == -1
    
    # Enhanced fields should have default values
    assert position.confidence == 0.0
    assert position.market_regime == "unknown"
    assert position.regime_stability == 0.0
    assert position.prediction_phase == "unknown"
    
    # Duration property should still work
    duration = position.duration_days
    expected_duration = 30.0 / (24 * 60)  # 30 minutes in days
    assert abs(duration - expected_duration) < 0.001
    
    print("✅ Backward compatibility - PASSED")


def test_position_duration_calculation():
    """Test duration calculation works with enhanced Position."""
    
    position = Position(
        open_time=pd.Timestamp('2025-01-01 09:00:00'),
        close_time=pd.Timestamp('2025-01-02 09:00:00'),  # 24 hours later
        size_usd=2000.0,
        pnl=100.0,
        direction=1,
        confidence=0.65,
        market_regime="medium_vol_trending",
        regime_stability=0.70,
        prediction_phase="CONTEXT_BOUNDARY"
    )
    
    # Duration should be 1.0 days
    assert abs(position.duration_days - 1.0) < 0.001
    
    print("✅ Duration calculation - PASSED")


def test_position_field_types():
    """Test that enhanced fields accept correct types."""
    
    position = Position(
        open_time=pd.Timestamp('2025-01-01 12:00:00'),
        close_time=pd.Timestamp('2025-01-01 12:45:00'),
        size_usd=750.0,
        pnl=0.0,
        direction=1,
        confidence=0.0,          # Edge case: zero confidence
        market_regime="unknown",  # Edge case: unknown regime
        regime_stability=1.0,     # Edge case: maximum stability
        prediction_phase="WARM_UP_PERIOD"
    )
    
    # Validate types
    assert isinstance(position.confidence, float)
    assert isinstance(position.market_regime, str)
    assert isinstance(position.regime_stability, float)
    assert isinstance(position.prediction_phase, str)
    
    # Validate bounds
    assert 0.0 <= position.confidence <= 1.0
    assert 0.0 <= position.regime_stability <= 1.0
    
    print("✅ Field types validation - PASSED")


def test_multiple_positions_creation():
    """Test creating multiple positions for batch processing."""
    
    positions = []
    
    for i in range(10):
        pos = Position(
            open_time=pd.Timestamp('2025-01-01 10:00:00') + pd.Timedelta(minutes=i*15),
            close_time=pd.Timestamp('2025-01-01 10:00:00') + pd.Timedelta(minutes=i*15+10),
            size_usd=1000.0 + i*100,
            pnl=np.random.normal(10.0, 5.0),
            direction=1 if i % 2 == 0 else -1,
            confidence=0.5 + i*0.05,
            market_regime=["low_vol_trending", "medium_vol_trending"][i % 2],
            regime_stability=0.7 + i*0.02,
            prediction_phase=["STABLE_WINDOW", "CONTEXT_BOUNDARY"][i % 2]
        )
        positions.append(pos)
    
    # Validate all positions created successfully
    assert len(positions) == 10
    
    # Validate field progression
    assert positions[0].confidence == 0.5
    assert positions[9].confidence == 0.95
    
    # Validate different regimes
    regimes = set(pos.market_regime for pos in positions)
    assert "low_vol_trending" in regimes
    assert "medium_vol_trending" in regimes
    
    print("✅ Multiple positions creation - PASSED")


def run_sluice_1a_unit_tests():
    """Run all Sluice 1A unit tests."""
    
    print("🔧 Running Sluice 1A: Position Dataclass Enhancement Unit Tests")
    
    test_enhanced_position_creation_with_new_fields()
    test_position_backward_compatibility()
    test_position_duration_calculation() 
    test_position_field_types()
    test_multiple_positions_creation()
    
    print(f"\n🎯 Sluice 1A Unit Tests COMPLETE")
    print(f"📊 Enhanced Position dataclass - VALIDATED") 
    print(f"🔄 Backward compatibility - MAINTAINED")
    print(f"🏗️ ODEB metadata fields - FUNCTIONAL")
    
    return True


if __name__ == "__main__":
    success = run_sluice_1a_unit_tests()
    exit(0 if success else 1)