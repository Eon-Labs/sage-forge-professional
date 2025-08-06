#!/usr/bin/env python3
"""
TiRex Prediction Boundary Validator
==================================

🎯 NAMED & TAMED: The CONTEXT_BOUNDARY Phenomenon

This utility helps users understand and validate TiRex prediction boundary behavior.
It solves the "22 vs 23 predictions" mystery by clearly documenting the off-by-one
boundary condition that affects all TiRex prediction counting.

Usage:
    from sage_forge.models.prediction_boundary_validator import validate_prediction_boundaries
    
    # Validate expected predictions for your bar sequence
    analysis = validate_prediction_boundaries(total_bars=150, sequence_length=128)
    print(f"Expected predictions: {analysis['expected_predictions']}")  # 23, not 22!
"""

from typing import Dict, List
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class PredictionBoundaryAnalysis:
    """Complete analysis of TiRex prediction boundary behavior."""
    total_bars: int
    sequence_length: int
    expected_predictions: int
    warm_up_bars: int
    first_prediction_bar: int  # 0-indexed
    last_prediction_bar: int   # 0-indexed
    boundary_formula: str
    phases: Dict[str, Dict]


def validate_prediction_boundaries(total_bars: int, sequence_length: int = 128) -> PredictionBoundaryAnalysis:
    """
    Validate and explain TiRex prediction boundary behavior.
    
    🎯 THE CONTEXT_BOUNDARY FORMULA:
    expected_predictions = total_bars - sequence_length + 1
    
    Args:
        total_bars: Total number of bars in your data sequence
        sequence_length: TiRex context window size (default 128)
        
    Returns:
        Complete boundary analysis with phase breakdown
        
    Example:
        >>> analysis = validate_prediction_boundaries(150, 128)
        >>> print(analysis.expected_predictions)  # 23 (not 22!)
        >>> print(analysis.first_prediction_bar)  # 127 (not 128!)
    """
    
    if total_bars < sequence_length:
        # All bars are in warm-up phase
        return PredictionBoundaryAnalysis(
            total_bars=total_bars,
            sequence_length=sequence_length,
            expected_predictions=0,
            warm_up_bars=total_bars,
            first_prediction_bar=-1,  # No predictions
            last_prediction_bar=-1,   # No predictions
            boundary_formula=f"{total_bars} - {sequence_length} + 1 = 0 (insufficient context)",
            phases={
                "WARM_UP_PERIOD": {
                    "bar_range": f"0-{total_bars-1}",
                    "bar_count": total_bars,
                    "description": "Insufficient context - no predictions possible"
                }
            }
        )
    
    # Calculate boundary points
    expected_predictions = total_bars - sequence_length + 1
    warm_up_bars = sequence_length - 1
    first_prediction_bar = sequence_length - 1  # 0-indexed
    last_prediction_bar = total_bars - 1
    
    # Define prediction phases
    phases = {
        "WARM_UP_PERIOD": {
            "bar_range": f"0-{warm_up_bars}",
            "bar_count": warm_up_bars + 1,
            "description": "Accumulating context - no predictions yet"
        },
        "CONTEXT_BOUNDARY": {
            "bar_range": f"{first_prediction_bar}",
            "bar_count": 1,
            "description": "First prediction with minimum context"
        }
    }
    
    if expected_predictions > 1:
        phases["STABLE_WINDOW"] = {
            "bar_range": f"{first_prediction_bar+1}-{last_prediction_bar}",
            "bar_count": expected_predictions - 1,
            "description": "Sliding window predictions"
        }
    
    return PredictionBoundaryAnalysis(
        total_bars=total_bars,
        sequence_length=sequence_length,
        expected_predictions=expected_predictions,
        warm_up_bars=warm_up_bars,
        first_prediction_bar=first_prediction_bar,
        last_prediction_bar=last_prediction_bar,
        boundary_formula=f"{total_bars} - {sequence_length} + 1 = {expected_predictions}",
        phases=phases
    )


def display_boundary_analysis(analysis: PredictionBoundaryAnalysis, show_examples: bool = True):
    """Display comprehensive boundary analysis with visual breakdown."""
    
    # Main analysis panel
    analysis_content = f"""
🎯 **PREDICTION BOUNDARY ANALYSIS**

**Input Parameters:**
• Total Bars: {analysis.total_bars}
• Sequence Length: {analysis.sequence_length}

**Boundary Formula:**
• expected_predictions = {analysis.boundary_formula}

**Critical Boundaries:**
• First Prediction Bar: {analysis.first_prediction_bar} (0-indexed)
• Last Prediction Bar: {analysis.last_prediction_bar} (0-indexed)
• Warm-up Bars: {analysis.warm_up_bars} bars
"""
    
    console.print(Panel(analysis_content, title="🔍 TiRex Prediction Boundaries", style="cyan"))
    
    # Phase breakdown table
    if analysis.phases:
        phase_table = Table(title="📊 Prediction Phase Breakdown", box=None)
        phase_table.add_column("Phase", style="bold cyan", min_width=16)
        phase_table.add_column("Bar Range", style="yellow", min_width=12)
        phase_table.add_column("Count", style="green", min_width=8)
        phase_table.add_column("Description", style="blue", min_width=30)
        
        for phase_name, phase_info in analysis.phases.items():
            phase_table.add_row(
                phase_name,
                phase_info["bar_range"],
                str(phase_info["bar_count"]),
                phase_info["description"]
            )
        
        console.print(phase_table)
    
    # Common examples
    if show_examples:
        examples_content = f"""
**Common Boundary Examples:**

• 150 bars, 128 context → **23 predictions** (bars 127-149)
• 100 bars, 128 context → **0 predictions** (insufficient context)
• 200 bars, 64 context → **137 predictions** (bars 63-199)

**❌ Common Mistakes:**
• Expecting 150-128=22 predictions (wrong!)
• Thinking first prediction at bar 128 (wrong!)
• Not accounting for 0-indexed bar counting (wrong!)

**✅ Correct Understanding:**
• Formula: total_bars - sequence_length + 1
• First prediction at bar (sequence_length - 1)
• CONTEXT_BOUNDARY phase is critically important
"""
        
        console.print(Panel(examples_content, title="📚 Boundary Examples & Common Mistakes", style="yellow"))


def validate_common_scenarios():
    """Validate common TiRex usage scenarios and display results."""
    
    console.print("\n🧪 **COMMON SCENARIO VALIDATION**")
    console.print("=" * 60)
    
    scenarios = [
        (150, 128, "Adversarial Test Scenario"),
        (1440, 128, "1 Day Minute Data"),
        (100, 128, "Insufficient Context"),
        (200, 64, "Smaller Context Window"),
        (10080, 128, "1 Week Minute Data")
    ]
    
    for total_bars, seq_len, description in scenarios:
        analysis = validate_prediction_boundaries(total_bars, seq_len)
        
        console.print(f"\n📊 **{description}**")
        console.print(f"   Bars: {total_bars}, Context: {seq_len}")
        console.print(f"   Expected Predictions: {analysis.expected_predictions}")
        
        if analysis.expected_predictions > 0:
            console.print(f"   First Prediction Bar: {analysis.first_prediction_bar}")
            console.print(f"   Boundary Formula: {analysis.boundary_formula}")
        else:
            console.print(f"   Status: Insufficient context for predictions")


if __name__ == "__main__":
    # Validate the specific scenario that caused the 22 vs 23 confusion
    console.print("🎯 **RESOLVING THE 22 vs 23 MYSTERY**")
    console.print("=" * 60)
    
    analysis = validate_prediction_boundaries(150, 128)
    display_boundary_analysis(analysis)
    
    # Validate common scenarios
    validate_common_scenarios()