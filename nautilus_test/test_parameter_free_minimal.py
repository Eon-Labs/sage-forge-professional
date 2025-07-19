#!/usr/bin/env python3
"""
Minimal test of parameter-free strategy to isolate performance issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.backtests.parameter_free_strategy_2025 import (
    TrulyParameterFreeStrategy,
    ClaSPInspiredChangePointDetector,
    FTRLOnlineLearner
)
from strategies.sota.enhanced_profitable_strategy_v2 import create_sota_strategy_config
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import BarType
from decimal import Decimal
import numpy as np

print("🧪 Minimal Parameter-Free Strategy Test")

# Create minimal config
try:
    strategy_config = create_sota_strategy_config(
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.SIM"),
        bar_type=BarType.from_str("BTCUSDT-PERP.SIM-1-MINUTE-LAST-EXTERNAL"),
        trade_size=Decimal("0.001"),
    )
    
    # Add fast mode
    strategy_config.fast_mode = True
    
    print("✅ Config created")
    
    # Test strategy creation
    strategy = TrulyParameterFreeStrategy(config=strategy_config)
    print("✅ Strategy created successfully")
    
    # Test core components individually
    print("\n🔧 Testing core components:")
    
    # Test ClaSP detector
    detector = ClaSPInspiredChangePointDetector()
    test_data = np.random.random(100)
    change_points = detector.detect_change_points(test_data)
    print(f"   ✅ ClaSP detector: Found {len(change_points)} change points")
    
    # Test FTRL learner
    ftrl = FTRLOnlineLearner(6)
    features = np.random.random(6)
    prediction = ftrl.predict(features)
    print(f"   ✅ FTRL learner: Prediction = {prediction:.4f}")
    
    print("\n🌟 All components working! Strategy should run faster now.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()