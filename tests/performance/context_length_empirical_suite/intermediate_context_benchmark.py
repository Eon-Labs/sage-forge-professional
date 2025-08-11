#!/usr/bin/env python3
"""
Intermediate TiRex Context Length Performance Benchmark
=======================================================

Focused testing with more context lengths to validate findings and explore
the performance sweet spot in detail.

Intermediate Test Matrix:
- Around proven sweet spot: [192, 240, 288, 336, 384, 480, 512, 640, 768]
- Validate 144 paradox and find optimal range
- Test memory scaling patterns
- Reasonable test duration while getting more insights

Goals:
1. Confirm 144 vs 288/512 paradox with more data points  
2. Find the exact optimal performance sweet spot
3. Test scaling between 512 and 1024 range
4. Provide actionable backtesting recommendations
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

import torch
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from rich import box

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message=".*TORCH_CUDA_ARCH_LIST.*")

console = Console()

# Import TiRex
try:
    from tirex import load_model
    TIREX_AVAILABLE = True
    console.print("✅ TiRex library available")
except ImportError as e:
    TIREX_AVAILABLE = False
    console.print(f"❌ TiRex library not available: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


@dataclass
class IntermediateBenchmarkResult:
    """Intermediate benchmark result with key metrics."""
    context_length: int
    avg_inference_ms: float
    gpu_memory_mb: float
    throughput_pred_sec: float
    successful_predictions: int
    total_attempts: int
    cuda_compilation_time: float
    relative_efficiency: float  # Performance relative to baseline
    memory_per_timestep: float  # MB per timestep for scaling analysis
    context_hours: float        # Context length in hours (5min bars)


class IntermediateContextBenchmark:
    """Intermediate context length benchmark focused on actionable insights."""
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        console.print(f"🔥 Using device: {self.device}")
        
        if self.device == "cpu":
            console.print("⚠️ Running on CPU - results will not be representative")
        else:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"🚀 GPU Memory: {gpu_memory_gb:.1f} GB available")
        
        self.results: List[IntermediateBenchmarkResult] = []
        self.baseline_288_ms = None  # Baseline from 288 timesteps
        
    def get_intermediate_test_contexts(self) -> List[int]:
        """Get intermediate test context lengths focused on sweet spot exploration."""
        return [
            144,   # Known slow (paradox validation)
            192,   # Between 144 and 288
            240,   # Closer to 288
            288,   # Proven fast baseline
            336,   # After 288
            384,   # Round number
            432,   # 1.5x 288
            480,   # Before 512
            512,   # Proven quality
            640,   # Between 512 and 768
            768,   # Larger context
            896,   # Before 1024
            1024   # Large context benchmark
        ]
    
    def generate_test_data(self, num_points: int = 5000) -> torch.Tensor:
        """Generate realistic test data for intermediate testing."""
        console.print(f"📊 Generating {num_points} realistic test data points")
        
        # Simplified but realistic price generation
        base_price = 45000.0
        
        # Create realistic returns with multiple time scales
        returns = np.random.normal(0, 0.01, num_points)
        
        # Add persistence (momentum)
        for i in range(1, len(returns)):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Add some trend and volatility clustering  
        trend_length = num_points // 20  # Change trend every ~250 bars
        for i in range(0, num_points, trend_length):
            end_idx = min(i + trend_length, num_points)
            trend = np.random.normal(0, 0.002)  # Random trend
            returns[i:end_idx] += trend
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        console.print(f"📈 Generated price range: ${prices.min():.0f} - ${prices.max():.0f}")
        return torch.tensor(prices, dtype=torch.float32)
    
    def measure_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            torch.cuda.synchronize()
            memory_bytes = torch.cuda.memory_allocated()
            return memory_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def benchmark_context_length(self, 
                                context_length: int,
                                test_data: torch.Tensor,
                                num_predictions: int = 20) -> IntermediateBenchmarkResult:
        """Benchmark specific context length with detailed metrics."""
        console.print(f"\n🔍 Benchmarking context: {context_length} timesteps")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load model with timing
        start_compile = time.perf_counter()
        try:
            model = load_model("NX-AI/TiRex", device=self.device)
            end_compile = time.perf_counter()
            cuda_compilation_time = end_compile - start_compile
            if cuda_compilation_time > 1.0:
                console.print(f"✅ Model loaded in {cuda_compilation_time:.1f}s")
        except Exception as e:
            console.print(f"❌ Failed to load model: {e}")
            return IntermediateBenchmarkResult(
                context_length=context_length,
                avg_inference_ms=float('inf'),
                gpu_memory_mb=0.0,
                throughput_pred_sec=0.0,
                successful_predictions=0,
                total_attempts=num_predictions,
                cuda_compilation_time=0.0,
                relative_efficiency=0.0,
                memory_per_timestep=float('inf'),
                context_hours=context_length * 5 / 60
            )
        
        # Verify sufficient data
        if len(test_data) < context_length + num_predictions:
            console.print(f"⚠️ Insufficient data: need {context_length + num_predictions}, have {len(test_data)}")
            return IntermediateBenchmarkResult(
                context_length=context_length,
                avg_inference_ms=float('inf'),
                gpu_memory_mb=0.0,
                throughput_pred_sec=0.0,
                successful_predictions=0,
                total_attempts=num_predictions,
                cuda_compilation_time=cuda_compilation_time,
                relative_efficiency=0.0,
                memory_per_timestep=float('inf'),
                context_hours=context_length * 5 / 60
            )
        
        # Performance tracking
        inference_times = []
        successful = 0
        gpu_memory_peak = 0.0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Context {context_length}", total=num_predictions)
            
            for i in range(num_predictions):
                try:
                    # Extract context window
                    start_idx = i
                    end_idx = start_idx + context_length
                    
                    if end_idx >= len(test_data):
                        break
                    
                    context_window = test_data[start_idx:end_idx].unsqueeze(0)  # [1, context_length]
                    
                    # Measure memory before
                    memory_before = self.measure_gpu_memory()
                    
                    # Time the inference
                    start_time = time.perf_counter()
                    
                    # Make prediction
                    quantiles, mean = model.forecast(
                        context_window,
                        prediction_length=1,
                        quantile_levels=[0.1, 0.5, 0.9]  # Safe quantiles
                    )
                    
                    end_time = time.perf_counter()
                    
                    # Measure memory after
                    memory_after = self.measure_gpu_memory()
                    gpu_memory_peak = max(gpu_memory_peak, memory_after)
                    
                    # Record successful inference
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                    successful += 1
                    
                    progress.update(task, advance=1)
                    
                except torch.cuda.OutOfMemoryError:
                    console.print(f"🚨 GPU OOM at prediction {i}")
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    logger.warning(f"Prediction {i} failed: {e}")
                    progress.update(task, advance=1)
                    continue
        
        # Calculate metrics
        avg_inference_ms = np.mean(inference_times) if inference_times else float('inf')
        throughput = 1000.0 / avg_inference_ms if avg_inference_ms != float('inf') else 0.0
        memory_per_timestep = gpu_memory_peak / context_length if context_length > 0 else float('inf')
        
        # Set/calculate baseline efficiency
        if context_length == 288:
            self.baseline_288_ms = avg_inference_ms
        
        relative_efficiency = (self.baseline_288_ms / avg_inference_ms) if (self.baseline_288_ms and avg_inference_ms != float('inf')) else 1.0
        
        context_hours = context_length * 5 / 60  # 5-minute bars to hours
        
        return IntermediateBenchmarkResult(
            context_length=context_length,
            avg_inference_ms=avg_inference_ms,
            gpu_memory_mb=gpu_memory_peak,
            throughput_pred_sec=throughput,
            successful_predictions=successful,
            total_attempts=num_predictions,
            cuda_compilation_time=cuda_compilation_time,
            relative_efficiency=relative_efficiency,
            memory_per_timestep=memory_per_timestep,
            context_hours=context_hours
        )
    
    def run_intermediate_benchmark_suite(self) -> List[IntermediateBenchmarkResult]:
        """Run intermediate benchmark suite."""
        console.print(Panel.fit(
            "[bold cyan]🔍 Intermediate TiRex Context Length Benchmark[/bold cyan]\n"
            "Focused testing around performance sweet spot\n"
            f"Hardware: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n"
            f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
            style="cyan"
        ))
        
        # Generate test data
        test_data = self.generate_test_data(num_points=5000)
        
        # Get test contexts
        context_lengths = self.get_intermediate_test_contexts()
        console.print(f"📊 Testing {len(context_lengths)} context lengths: {context_lengths}")
        
        start_time = time.perf_counter()
        
        # Run benchmarks
        for context_length in context_lengths:
            result = self.benchmark_context_length(
                context_length=context_length,
                test_data=test_data,
                num_predictions=20
            )
            self.results.append(result)
            
            # Brief pause between tests
            time.sleep(1.5)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        console.print(f"\n⏱️ Total benchmark duration: {total_duration/60:.1f} minutes")
        
        return self.results
    
    def display_intermediate_results(self):
        """Display intermediate results with detailed analysis."""
        if not self.results:
            console.print("❌ No results to display")
            return
        
        # Create detailed results table
        table = Table(title="📊 Intermediate Context Length Performance Analysis", box=box.ROUNDED)
        table.add_column("Context", style="cyan", min_width=8)
        table.add_column("Hours", style="dim", min_width=6)
        table.add_column("Time (ms)", style="bold", min_width=10)
        table.add_column("Memory", style="green", min_width=10)
        table.add_column("Efficiency", style="yellow", min_width=10)
        table.add_column("Throughput", style="blue", min_width=12)
        table.add_column("Status", style="magenta", min_width=12)
        
        valid_results = [r for r in self.results if r.avg_inference_ms != float('inf')]
        
        # Find key performers
        fastest = min(valid_results, key=lambda x: x.avg_inference_ms) if valid_results else None
        most_efficient = max(valid_results, key=lambda x: x.relative_efficiency) if valid_results else None
        
        for result in self.results:
            if result.avg_inference_ms == float('inf'):
                time_str = "FAILED"
                memory_str = "N/A"
                efficiency_str = "N/A"
                throughput_str = "N/A"
                status = "❌ FAILED"
            else:
                time_str = f"{result.avg_inference_ms:.1f}"
                memory_str = f"{result.gpu_memory_mb:.0f} MB"
                efficiency_str = f"{result.relative_efficiency:.2f}x"
                throughput_str = f"{result.throughput_pred_sec:.1f}/sec"
                
                status = ""
                if result == fastest:
                    status = "⚡ FASTEST"
                elif result == most_efficient:
                    status = "🏆 OPTIMAL"
                elif result.context_length == 144:
                    status = "🐌 PARADOX"
                elif result.relative_efficiency >= 1.0:
                    status = "✅ FAST"
                else:
                    status = "📈 SLOWER"
            
            table.add_row(
                f"{result.context_length}",
                f"{result.context_hours:.1f}h",
                time_str,
                memory_str,
                efficiency_str,
                throughput_str,
                status
            )
        
        console.print(table)
        
        # Analysis and insights
        self._analyze_intermediate_patterns(valid_results)
    
    def _analyze_intermediate_patterns(self, valid_results: List[IntermediateBenchmarkResult]):
        """Analyze patterns from intermediate results."""
        if not valid_results:
            console.print("❌ No valid results for analysis")
            return
        
        console.print(f"\n🔬 **INTERMEDIATE PERFORMANCE ANALYSIS**")
        
        # Find key metrics
        fastest = min(valid_results, key=lambda x: x.avg_inference_ms)
        slowest = max(valid_results, key=lambda x: x.avg_inference_ms)
        most_efficient = max(valid_results, key=lambda x: x.relative_efficiency)
        
        # Check for 144 paradox
        result_144 = next((r for r in valid_results if r.context_length == 144), None)
        result_288 = next((r for r in valid_results if r.context_length == 288), None)
        
        console.print(f"\n⚡ **Speed Analysis:**")
        console.print(f"• **Fastest**: {fastest.context_length} timesteps ({fastest.avg_inference_ms:.1f}ms)")
        console.print(f"• **Slowest**: {slowest.context_length} timesteps ({slowest.avg_inference_ms:.1f}ms)")
        console.print(f"• **Speed Range**: {fastest.avg_inference_ms:.1f}ms - {slowest.avg_inference_ms:.1f}ms ({slowest.avg_inference_ms/fastest.avg_inference_ms:.1f}x)")
        
        if result_144 and result_288:
            console.print(f"• **144 vs 288 Paradox**: {'CONFIRMED' if result_144.avg_inference_ms > result_288.avg_inference_ms else 'NOT CONFIRMED'}")
            console.print(f"  - 144 timesteps: {result_144.avg_inference_ms:.1f}ms")
            console.print(f"  - 288 timesteps: {result_288.avg_inference_ms:.1f}ms")
        
        console.print(f"\n🏆 **Efficiency Analysis:**")
        console.print(f"• **Most Efficient**: {most_efficient.context_length} timesteps ({most_efficient.relative_efficiency:.2f}x baseline)")
        
        # Memory scaling analysis
        console.print(f"\n📊 **Memory Scaling:**")
        context_lengths = [r.context_length for r in valid_results]
        memory_usage = [r.gpu_memory_mb for r in valid_results]
        
        if len(valid_results) >= 2:
            # Simple linear fit
            memory_per_context = (memory_usage[-1] - memory_usage[0]) / (context_lengths[-1] - context_lengths[0])
            console.print(f"• **Scaling Rate**: {memory_per_context:.3f} MB per timestep")
            console.print(f"• **Memory Range**: {min(memory_usage):.0f}MB - {max(memory_usage):.0f}MB")
            
            # Predict memory for larger contexts
            predicted_2048 = memory_usage[0] + (2048 - context_lengths[0]) * memory_per_context
            predicted_4096 = memory_usage[0] + (4096 - context_lengths[0]) * memory_per_context
            console.print(f"• **Predicted 2048**: {predicted_2048:.0f}MB")
            console.print(f"• **Predicted 4096**: {predicted_4096:.0f}MB")
        
        # Performance sweet spot identification
        self._identify_sweet_spots(valid_results)
    
    def _identify_sweet_spots(self, valid_results: List[IntermediateBenchmarkResult]):
        """Identify performance sweet spots for different use cases."""
        console.print(f"\n🎯 **PERFORMANCE SWEET SPOTS**")
        
        # Sort by efficiency
        by_efficiency = sorted(valid_results, key=lambda x: x.relative_efficiency, reverse=True)
        top_3_efficient = by_efficiency[:3]
        
        # Sort by speed
        by_speed = sorted(valid_results, key=lambda x: x.avg_inference_ms)
        top_3_fast = by_speed[:3]
        
        console.print(f"\n⚡ **Top 3 Fastest:**")
        for i, result in enumerate(top_3_fast, 1):
            console.print(f"   {i}. {result.context_length} timesteps: {result.avg_inference_ms:.1f}ms ({result.context_hours:.1f}h context)")
        
        console.print(f"\n🏆 **Top 3 Most Efficient:**")  
        for i, result in enumerate(top_3_efficient, 1):
            console.print(f"   {i}. {result.context_length} timesteps: {result.relative_efficiency:.2f}x baseline ({result.context_hours:.1f}h context)")
        
        # Recommendations
        console.print(f"\n📋 **BACKTESTING RECOMMENDATIONS:**")
        
        fastest = by_speed[0]
        most_efficient = by_efficiency[0]
        quality_context = max(valid_results, key=lambda x: x.context_length)
        
        console.print(f"• **Rapid Development**: Use **{fastest.context_length} timesteps** ({fastest.avg_inference_ms:.1f}ms per prediction)")
        console.print(f"• **Optimal Balance**: Use **{most_efficient.context_length} timesteps** ({most_efficient.relative_efficiency:.2f}x efficiency)")
        console.print(f"• **Quality Focus**: Use **{quality_context.context_length} timesteps** ({quality_context.context_hours:.1f}h context)")
        
        # Scenario analysis
        console.print(f"\n⏰ **Backtesting Time Estimates:**")
        scenarios = [
            ("Quick Test (1,000 predictions)", 1000),
            ("Medium Backtest (10,000 predictions)", 10000),
            ("Comprehensive Backtest (100,000 predictions)", 100000)
        ]
        
        for scenario_name, num_preds in scenarios:
            console.print(f"\n   {scenario_name}:")
            for context in [fastest.context_length, most_efficient.context_length, quality_context.context_length]:
                result = next(r for r in valid_results if r.context_length == context)
                total_seconds = (result.avg_inference_ms * num_preds) / 1000
                if total_seconds < 60:
                    time_str = f"{total_seconds:.1f}s"
                elif total_seconds < 3600:
                    time_str = f"{total_seconds/60:.1f}min"
                else:
                    time_str = f"{total_seconds/3600:.1f}h"
                console.print(f"   • {context} timesteps: {time_str}")
    
    def export_intermediate_results(self, filepath: str = None):
        """Export intermediate results."""
        if not self.results:
            console.print("❌ No results to export")
            return
        
        if filepath is None:
            filepath = f"intermediate_context_benchmark_{int(time.time())}.csv"
        
        # Prepare data
        data = []
        for result in self.results:
            data.append({
                'context_length': result.context_length,
                'context_hours_5min_bars': result.context_hours,
                'avg_inference_ms': result.avg_inference_ms,
                'gpu_memory_mb': result.gpu_memory_mb,
                'throughput_pred_sec': result.throughput_pred_sec,
                'successful_predictions': result.successful_predictions,
                'total_attempts': result.total_attempts,
                'success_rate_pct': (result.successful_predictions / result.total_attempts * 100) if result.total_attempts > 0 else 0,
                'cuda_compilation_time': result.cuda_compilation_time,
                'relative_efficiency_vs_288': result.relative_efficiency,
                'memory_per_timestep_mb': result.memory_per_timestep,
                'feasible': result.avg_inference_ms != float('inf')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        console.print(f"📄 Intermediate results exported to: {filepath}")


def main():
    """Main execution for intermediate context benchmark."""
    console.print("[bold green]🔍 Intermediate TiRex Context Length Benchmark[/bold green]")
    
    try:
        # Initialize benchmark
        benchmark = IntermediateContextBenchmark()
        
        # Run intermediate benchmark
        benchmark.run_intermediate_benchmark_suite()
        
        # Display results with analysis
        benchmark.display_intermediate_results()
        
        # Export results
        results_dir = Path("/home/tca/eon/nt/tests/performance/context_length_empirical_suite/results")
        results_dir.mkdir(exist_ok=True)
        
        export_path = results_dir / f"intermediate_benchmark_{int(time.time())}.csv"
        benchmark.export_intermediate_results(str(export_path))
        
        console.print(f"\n✅ Intermediate benchmark complete! Results: {export_path}")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Intermediate benchmark interrupted")
        return 1
    except Exception as e:
        console.print(f"❌ Intermediate benchmark failed: {e}")
        logger.exception("Intermediate benchmark failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)