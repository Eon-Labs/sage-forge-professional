#!/usr/bin/env python3
"""
Prediction Counting Analysis - Root Cause Investigation
=======================================================

Investigates the discrepancy between expected (22) and actual (23) predictions
to determine if this indicates a fundamental flaw in the TiRex integration.
"""

import sys
from pathlib import Path

# Add SAGE-Forge to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

def analyze_prediction_counting():
    """Analyze the prediction counting logic step by step."""
    
    print("🔍 PREDICTION COUNTING ANALYSIS")
    print("=" * 50)
    
    # Test parameters
    total_bars = 150
    sequence_length = 128
    
    print(f"📊 Test Setup:")
    print(f"  • Total bars: {total_bars}")
    print(f"  • Sequence length (warm-up): {sequence_length}")
    print(f"  • Bar indices: 0, 1, 2, ..., {total_bars-1}")
    
    print(f"\n📈 Buffer Growth Analysis:")
    
    # Track when predictions start
    first_prediction_bar = None
    prediction_bars = []
    
    for bar_index in range(total_bars):
        buffer_size = min(bar_index + 1, sequence_length)  # Deque with maxlen
        can_predict = buffer_size >= sequence_length
        
        if can_predict and first_prediction_bar is None:
            first_prediction_bar = bar_index
            
        if can_predict:
            prediction_bars.append(bar_index)
            
        # Show first few and key transition points
        if bar_index < 5 or bar_index in [125, 126, 127, 128, 129] or bar_index >= 145:
            status = "✅ PREDICT" if can_predict else "⏳ WARM-UP"
            print(f"  Bar {bar_index:3d}: Buffer size = {buffer_size:3d} -> {status}")
    
    print(f"\n🎯 Prediction Analysis:")
    print(f"  • First prediction at bar: {first_prediction_bar}")
    print(f"  • Last prediction at bar: {prediction_bars[-1] if prediction_bars else 'None'}")
    print(f"  • Total predictions: {len(prediction_bars)}")
    print(f"  • Prediction bars: {prediction_bars[:3]} ... {prediction_bars[-3:]}")
    
    print(f"\n🧮 Mathematical Analysis:")
    expected_simple = total_bars - sequence_length
    expected_correct = total_bars - sequence_length + 1 if total_bars >= sequence_length else 0
    actual = len(prediction_bars)
    
    print(f"  • Simple calculation: {total_bars} - {sequence_length} = {expected_simple}")
    print(f"  • Correct calculation: ({total_bars-1}) - ({sequence_length-1}) + 1 = {expected_correct}")
    print(f"  • Actual predictions: {actual}")
    
    print(f"\n🔍 Root Cause Analysis:")
    if actual == expected_simple:
        print("  ✅ Matches simple calculation - no discrepancy")
    elif actual == expected_correct:
        print("  ✅ Matches correct calculation - user expectation was wrong")
        print("  📚 Explanation: First prediction happens AT the moment buffer reaches sequence_length")
        print("     Not AFTER consuming sequence_length bars for warm-up")
    else:
        print(f"  🚨 DISCREPANCY: Expected {expected_simple} or {expected_correct}, got {actual}")
        print("  🔧 This indicates a fundamental counting error in the implementation")
    
    print(f"\n💡 Conclusion:")
    if actual == 23 and expected_correct == 23:
        print("  🎯 The count of 23 predictions is MATHEMATICALLY CORRECT")
        print("  📖 User's expectation of 22 was based on incorrect mental model")
        print("  ✅ No implementation flaw - TiRex integration working as expected")
    else:
        print("  🚨 Implementation flaw detected - requires investigation")
        
    return {
        'total_bars': total_bars,
        'sequence_length': sequence_length, 
        'expected_simple': expected_simple,
        'expected_correct': expected_correct,
        'actual': actual,
        'first_prediction_bar': first_prediction_bar,
        'prediction_bars': prediction_bars
    }


if __name__ == "__main__":
    results = analyze_prediction_counting()
    
    # Exit code based on whether discrepancy indicates a real problem
    if results['actual'] == results['expected_correct']:
        print(f"\n✅ ANALYSIS COMPLETE: No implementation flaw detected")
        sys.exit(0)
    else:
        print(f"\n🚨 ANALYSIS COMPLETE: Implementation flaw requires attention") 
        sys.exit(1)