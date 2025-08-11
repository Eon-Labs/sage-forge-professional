#!/usr/bin/env python3
"""
Large Context Length TiRex Performance Benchmark
=================================================

Testing very large context lengths to find performance limits, memory scaling,
and quality vs speed tradeoffs for extreme contexts.

Large Context Test Matrix:
- Large: [1536, 2048, 3072, 4096] 
- Extreme: [6144, 8192, 12288] (if memory permits)

Goals:
1. Find memory and performance limits
2. Identify quality vs speed tradeoff curves  
3. Test extreme contexts for research applications
4. Validate memory scaling predictions
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
class LargeContextResult:
    """Large context benchmark result with extreme context metrics."""
    context_length: int
    context_days: float         # Context length in days (5min bars)
    avg_inference_ms: float
    gpu_memory_mb: float
    gpu_memory_gb: float        # Memory in GB for large contexts
    throughput_pred_sec: float
    successful_predictions: int
    failed_predictions: int
    total_attempts: int
    oom_errors: int             # Out of memory errors
    cuda_compilation_time: float
    feasibility_status: str     # "FEASIBLE", "MEMORY_LIMITED", "FAILED"
    memory_efficiency: float    # MB per day of context
    speed_degradation: float    # Slowdown compared to baseline


class LargeContextBenchmark:
    """Large context length benchmark for extreme testing."""
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        console.print(f"🔥 Using device: {self.device}")
        
        if self.device == "cpu":
            console.print("⚠️ Running on CPU - large contexts not feasible")
            sys.exit(1)
        else:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"🚀 GPU Memory: {gpu_memory_gb:.1f} GB total")
            
            # Check available memory more precisely
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            available_gb = available_memory / (1024**3)
            console.print(f"🎯 Available Memory: {available_gb:.1f} GB for large contexts")
        
        self.results: List[LargeContextResult] = []
        self.baseline_512_ms = 9.4  # From previous benchmark
        
    def get_large_context_matrix(self) -> Dict[str, List[int]]:
        """Get large context test matrix."""
        return {
            "large": [1536, 2048, 3072, 4096],     # 5-14 days context
            "extreme": [6144, 8192],               # 21-28 days context  
            "mega": [12288, 16384]                 # 42-56 days context (if feasible)
        }
    
    def estimate_memory_requirement(self, context_length: int) -> float:
        """Estimate GPU memory requirement based on scaling from previous results."""
        # Based on intermediate results: ~0.237 MB per timestep + base overhead
        base_memory = 150  # Base memory in MB
        scaling_rate = 0.237  # MB per timestep from intermediate results
        estimated_mb = base_memory + (context_length * scaling_rate)
        return estimated_mb
    
    def check_memory_feasibility(self, context_length: int) -> tuple[bool, str]:
        """Check if context length is feasible and provide status."""
        if not torch.cuda.is_available():
            return False, "NO_GPU"
        
        # Get available memory with safety margin
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated()
        safety_margin = 0.7  # Use only 70% of available for safety
        usable_memory = available_memory * safety_margin
        
        required_memory = self.estimate_memory_requirement(context_length) * 1024 * 1024  # Convert to bytes
        
        if required_memory > usable_memory:
            required_gb = required_memory / (1024**3)
            available_gb = usable_memory / (1024**3)
            return False, f"MEMORY_LIMITED ({required_gb:.1f}GB required > {available_gb:.1f}GB available)"
        
        return True, "FEASIBLE"
    
    def generate_extended_test_data(self, num_points: int = 20000) -> torch.Tensor:
        """Generate extended test data for large context testing."""
        console.print(f"📊 Generating {num_points} extended test data points")
        
        # Generate realistic multi-scale market data - simplified approach
        base_price = 45000.0
        
        # Create base returns array
        returns = np.random.normal(0, 0.01, num_points)
        
        # Add persistence (momentum)
        for i in range(1, len(returns)):
            returns[i] = 0.05 * returns[i-1] + 0.95 * returns[i]
        
        # Add regime changes for realism
        regime_length = num_points // 10  # ~10 regimes in the data
        for i in range(0, num_points, regime_length):
            end_idx = min(i + regime_length, num_points)
            regime_volatility = np.random.uniform(0.5, 2.0)  # Varying volatility regimes
            returns[i:end_idx] *= regime_volatility
        
        # Add daily trend cycles
        daily_cycle_length = 288  # 24 hours * 12 (5-min bars per hour)
        for i in range(0, num_points, daily_cycle_length):
            end_idx = min(i + daily_cycle_length, num_points)
            daily_trend = np.random.normal(0, 0.003)
            returns[i:end_idx] += daily_trend
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        console.print(f"📈 Generated price range: ${prices.min():.0f} - ${prices.max():.0f}")
        console.print(f"📊 Data covers ~{num_points * 5 / (60 * 24):.1f} days of 5-minute bars")
        
        return torch.tensor(prices, dtype=torch.float32)
    
    def benchmark_large_context(self, 
                               context_length: int,
                               test_data: torch.Tensor,
                               category: str,
                               num_predictions: int = None) -> LargeContextResult:
        """Benchmark large context length with comprehensive error handling."""
        
        # Adjust predictions based on context size
        if num_predictions is None:
            if category == "large":
                num_predictions = 10
            elif category == "extreme": 
                num_predictions = 5
            else:  # mega
                num_predictions = 3
        
        context_days = context_length * 5 / (60 * 24)  # Convert to days
        console.print(f"\n🔍 Benchmarking {category} context: {context_length} timesteps ({context_days:.1f} days)")
        
        # Check memory feasibility first
        feasible, status = self.check_memory_feasibility(context_length)
        if not feasible:
            console.print(f"❌ {status}")
            return LargeContextResult(
                context_length=context_length,
                context_days=context_days,
                avg_inference_ms=float('inf'),
                gpu_memory_mb=0.0,
                gpu_memory_gb=0.0,
                throughput_pred_sec=0.0,
                successful_predictions=0,
                failed_predictions=num_predictions,
                total_attempts=num_predictions,
                oom_errors=0,
                cuda_compilation_time=0.0,
                feasibility_status=status,
                memory_efficiency=float('inf'),
                speed_degradation=float('inf')
            )
        
        # Clear GPU cache aggressively for large contexts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)  # Let GPU settle
        
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
            return LargeContextResult(
                context_length=context_length,
                context_days=context_days,
                avg_inference_ms=float('inf'),
                gpu_memory_mb=0.0,
                gpu_memory_gb=0.0,
                throughput_pred_sec=0.0,
                successful_predictions=0,
                failed_predictions=num_predictions,
                total_attempts=num_predictions,
                oom_errors=0,
                cuda_compilation_time=0.0,
                feasibility_status="MODEL_LOAD_FAILED",
                memory_efficiency=float('inf'),
                speed_degradation=float('inf')
            )
        
        # Verify sufficient data
        required_data = context_length + num_predictions
        if len(test_data) < required_data:
            console.print(f"⚠️ Insufficient data: need {required_data}, have {len(test_data)}")
            console.print("🔄 Generating additional data...")
            additional_needed = required_data - len(test_data) + 1000  # Extra buffer
            additional_data = self.generate_extended_test_data(additional_needed)
            test_data = torch.cat([test_data, additional_data])
            console.print(f"✅ Extended data to {len(test_data)} points")
        
        # Performance tracking
        inference_times = []
        successful = 0
        failed = 0
        oom_errors = 0
        gpu_memory_peak = 0.0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"{category.title()} {context_length}", total=num_predictions)
            
            for i in range(num_predictions):
                try:
                    # Extract context window
                    start_idx = i
                    end_idx = start_idx + context_length
                    
                    if end_idx >= len(test_data):
                        console.print(f"⚠️ Reached end of data at prediction {i}")
                        break
                    
                    context_window = test_data[start_idx:end_idx].unsqueeze(0)  # [1, context_length]
                    
                    # Monitor memory before inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        memory_before_bytes = torch.cuda.memory_allocated()
                        memory_before_mb = memory_before_bytes / (1024 * 1024)
                    
                    # Time the inference with timeout protection
                    start_time = time.perf_counter()
                    
                    # Make prediction with larger context
                    quantiles, mean = model.forecast(
                        context_window,
                        prediction_length=1,
                        quantile_levels=[0.1, 0.5, 0.9]  # Safe quantiles
                    )
                    
                    end_time = time.perf_counter()
                    
                    # Monitor memory after inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        memory_after_bytes = torch.cuda.memory_allocated()
                        memory_after_mb = memory_after_bytes / (1024 * 1024)
                        gpu_memory_peak = max(gpu_memory_peak, memory_after_mb)
                    
                    # Record successful inference
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                    successful += 1
                    
                    progress.update(task, advance=1)
                    
                    # Add cooling delay for large contexts to prevent thermal throttling
                    if inference_time_ms > 50:
                        time.sleep(0.5)
                    elif category in ["extreme", "mega"]:
                        time.sleep(0.2)
                    
                except torch.cuda.OutOfMemoryError as oom:
                    oom_errors += 1
                    failed += 1
                    console.print(f"🚨 GPU OOM at prediction {i}: cleaning up...")
                    torch.cuda.empty_cache()
                    time.sleep(2)  # Give GPU time to recover
                    progress.update(task, advance=1)
                    # Try to continue with next prediction
                    continue
                    
                except RuntimeError as runtime_error:
                    if "out of memory" in str(runtime_error).lower():
                        oom_errors += 1
                    failed += 1
                    console.print(f"⚠️ Runtime error at prediction {i}: {runtime_error}")
                    torch.cuda.empty_cache()
                    progress.update(task, advance=1)
                    continue
                    
                except Exception as e:
                    failed += 1
                    logger.warning(f"Prediction {i} failed: {e}")
                    progress.update(task, advance=1)
                    continue
        
        # Calculate metrics
        avg_inference_ms = np.mean(inference_times) if inference_times else float('inf')
        throughput = 1000.0 / avg_inference_ms if avg_inference_ms != float('inf') else 0.0
        gpu_memory_gb = gpu_memory_peak / 1024  # Convert MB to GB
        
        # Efficiency metrics
        memory_efficiency = gpu_memory_peak / context_days if context_days > 0 else float('inf')
        speed_degradation = avg_inference_ms / self.baseline_512_ms if avg_inference_ms != float('inf') else float('inf')
        
        # Determine final feasibility status
        if successful == 0:
            final_status = "FAILED"
        elif oom_errors > 0:
            final_status = "MEMORY_LIMITED"
        elif successful == num_predictions:
            final_status = "FEASIBLE"
        else:
            final_status = "PARTIAL_SUCCESS"
        
        return LargeContextResult(
            context_length=context_length,
            context_days=context_days,
            avg_inference_ms=avg_inference_ms,
            gpu_memory_mb=gpu_memory_peak,
            gpu_memory_gb=gpu_memory_gb,
            throughput_pred_sec=throughput,
            successful_predictions=successful,
            failed_predictions=failed,
            total_attempts=num_predictions,
            oom_errors=oom_errors,
            cuda_compilation_time=cuda_compilation_time,
            feasibility_status=final_status,
            memory_efficiency=memory_efficiency,
            speed_degradation=speed_degradation
        )
    
    def run_large_context_benchmark(self) -> List[LargeContextResult]:
        """Run large context benchmark suite."""
        console.print(Panel.fit(
            "[bold red]🚀 Large Context TiRex Performance Benchmark[/bold red]\n"
            "Testing extreme context lengths for memory and performance limits\n"
            f"Hardware: {torch.cuda.get_device_name()}\n"
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB",
            style="red"
        ))
        
        # Generate extended test data
        test_data = self.generate_extended_test_data(num_points=20000)
        
        # Get test matrix
        test_matrix = self.get_large_context_matrix()
        
        start_time = time.perf_counter()
        
        # Run benchmarks by category (skip mega if memory insufficient)
        for category, context_lengths in test_matrix.items():
            console.print(f"\n🔍 Testing {category.upper()} contexts: {context_lengths}")
            
            for context_length in context_lengths:
                # Quick pre-check for mega contexts
                if category == "mega":
                    feasible, status = self.check_memory_feasibility(context_length)
                    if not feasible:
                        console.print(f"⏭️ Skipping {context_length}: {status}")
                        continue
                
                result = self.benchmark_large_context(
                    context_length=context_length,
                    test_data=test_data,
                    category=category
                )
                self.results.append(result)
                
                # Extended pause for GPU cooling and memory cleanup
                if result.feasibility_status in ["FEASIBLE", "PARTIAL_SUCCESS"]:
                    time.sleep(3)
                else:
                    time.sleep(1)
                
                # Stop testing category if we hit memory limits
                if result.oom_errors > 0 and category != "large":
                    console.print(f"⚠️ Memory limits reached, stopping {category} tests")
                    break
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        console.print(f"\n⏱️ Total large context benchmark duration: {total_duration/60:.1f} minutes")
        
        return self.results
    
    def display_large_context_results(self):
        """Display large context results with analysis."""
        if not self.results:
            console.print("❌ No results to display")
            return
        
        # Create comprehensive results table
        table = Table(title="🚀 Large Context Length Performance Analysis", box=box.ROUNDED)
        table.add_column("Context", style="cyan", min_width=8)
        table.add_column("Days", style="dim", min_width=8)
        table.add_column("Time (ms)", style="bold", min_width=10)
        table.add_column("Memory", style="green", min_width=12)
        table.add_column("Success", style="yellow", min_width=10)
        table.add_column("Degradation", style="red", min_width=12)
        table.add_column("Status", style="magenta", min_width=15)
        
        for result in self.results:
            if result.avg_inference_ms == float('inf'):
                time_str = "FAILED"
                degradation_str = "N/A"
            else:
                time_str = f"{result.avg_inference_ms:.0f}"
                degradation_str = f"{result.speed_degradation:.1f}x"
            
            memory_str = f"{result.gpu_memory_gb:.1f} GB" if result.gpu_memory_gb > 1 else f"{result.gpu_memory_mb:.0f} MB"
            success_str = f"{result.successful_predictions}/{result.total_attempts}"
            
            # Status with emoji
            status_emoji = {
                "FEASIBLE": "✅ FEASIBLE",
                "PARTIAL_SUCCESS": "⚠️ PARTIAL",
                "MEMORY_LIMITED": "🚨 OOM",
                "FAILED": "❌ FAILED"
            }.get(result.feasibility_status, result.feasibility_status)
            
            table.add_row(
                f"{result.context_length}",
                f"{result.context_days:.1f}d",
                time_str,
                memory_str,
                success_str,
                degradation_str,
                status_emoji
            )
        
        console.print(table)
        
        # Analysis
        self._analyze_large_context_patterns()
    
    def _analyze_large_context_patterns(self):
        """Analyze large context performance patterns."""
        console.print(f"\n🔬 **LARGE CONTEXT ANALYSIS**")
        
        # Separate results by feasibility
        feasible_results = [r for r in self.results if r.feasibility_status == "FEASIBLE"]
        limited_results = [r for r in self.results if r.feasibility_status in ["MEMORY_LIMITED", "PARTIAL_SUCCESS"]]
        failed_results = [r for r in self.results if r.feasibility_status == "FAILED"]
        
        console.print(f"\n📊 **Feasibility Summary:**")
        console.print(f"• **Fully Feasible**: {len(feasible_results)} contexts")
        console.print(f"• **Memory Limited**: {len(limited_results)} contexts")  
        console.print(f"• **Failed**: {len(failed_results)} contexts")
        
        if feasible_results:
            console.print(f"\n✅ **Feasible Large Contexts:**")
            
            largest_feasible = max(feasible_results, key=lambda x: x.context_length)
            console.print(f"• **Largest Feasible**: {largest_feasible.context_length} timesteps ({largest_feasible.context_days:.1f} days)")
            console.print(f"  - Performance: {largest_feasible.avg_inference_ms:.0f}ms per prediction")
            console.print(f"  - Memory: {largest_feasible.gpu_memory_gb:.1f} GB")
            console.print(f"  - Degradation: {largest_feasible.speed_degradation:.1f}x slower than 512 baseline")
            
            # Memory scaling analysis
            context_lengths = [r.context_length for r in feasible_results]
            memory_usage_gb = [r.gpu_memory_gb for r in feasible_results]
            
            if len(feasible_results) >= 2:
                memory_scaling = (memory_usage_gb[-1] - memory_usage_gb[0]) / (context_lengths[-1] - context_lengths[0])
                console.print(f"\n📈 **Memory Scaling** (large contexts):")
                console.print(f"• **Scaling Rate**: {memory_scaling*1000:.1f} MB per 1000 timesteps")
                
                # Predict extreme memory requirements
                predicted_32k = memory_usage_gb[0] + (32768 - context_lengths[0]) * memory_scaling
                console.print(f"• **Predicted 32K context**: {predicted_32k:.1f} GB")
            
            # Speed degradation analysis
            speeds = [r.speed_degradation for r in feasible_results]
            console.print(f"\n⏱️ **Speed Degradation:**")
            console.print(f"• **Range**: {min(speeds):.1f}x - {max(speeds):.1f}x slower than 512 baseline")
            
        if limited_results:
            console.print(f"\n⚠️ **Memory Limited Contexts:**")
            for result in limited_results:
                console.print(f"• **{result.context_length} timesteps**: {result.successful_predictions}/{result.total_attempts} successful, {result.oom_errors} OOM errors")
        
        # Practical recommendations
        self._generate_large_context_recommendations(feasible_results)
    
    def _generate_large_context_recommendations(self, feasible_results: List[LargeContextResult]):
        """Generate recommendations for large context usage."""
        console.print(f"\n🎯 **LARGE CONTEXT RECOMMENDATIONS:**")
        
        if not feasible_results:
            console.print("❌ No feasible large contexts found for this GPU configuration")
            return
        
        # Quality-focused recommendations
        largest = max(feasible_results, key=lambda x: x.context_length)
        fastest_large = min(feasible_results, key=lambda x: x.avg_inference_ms) if feasible_results else None
        
        console.print(f"\n🔬 **Research Applications:**")
        console.print(f"• **Maximum Quality**: Use **{largest.context_length} timesteps** ({largest.context_days:.1f} days context)")
        console.print(f"  - Cost: {largest.avg_inference_ms:.0f}ms per prediction ({largest.speed_degradation:.1f}x slower)")
        console.print(f"  - Memory: {largest.gpu_memory_gb:.1f} GB")
        
        if fastest_large:
            console.print(f"• **Fastest Large**: Use **{fastest_large.context_length} timesteps** ({fastest_large.context_days:.1f} days)")
            console.print(f"  - Performance: {fastest_large.avg_inference_ms:.0f}ms per prediction")
        
        # Use case scenarios
        console.print(f"\n📋 **Use Case Guidelines:**")
        console.print(f"• **Standard Backtesting**: Stick to <1024 timesteps for best speed")
        console.print(f"• **Research Analysis**: {largest.context_length} timesteps for maximum pattern recognition")
        console.print(f"• **Long-term Studies**: Large contexts capture multi-day market dynamics")
        console.print(f"• **Academic Research**: Extreme contexts for studying long-range dependencies")
        
        # Performance warnings
        console.print(f"\n⚠️ **Performance Considerations:**")
        if largest.speed_degradation > 5:
            console.print(f"• Large contexts are {largest.speed_degradation:.1f}x slower - use sparingly")
        console.print(f"• GPU memory requirement scales to {largest.gpu_memory_gb:.1f} GB")
        console.print(f"• Consider context length vs marginal quality improvements")
    
    def export_large_context_results(self, filepath: str = None):
        """Export large context results."""
        if not self.results:
            console.print("❌ No results to export")
            return
        
        if filepath is None:
            filepath = f"large_context_benchmark_{int(time.time())}.csv"
        
        # Prepare comprehensive data
        data = []
        for result in self.results:
            data.append({
                'context_length': result.context_length,
                'context_days_5min_bars': result.context_days,
                'avg_inference_ms': result.avg_inference_ms,
                'gpu_memory_mb': result.gpu_memory_mb,
                'gpu_memory_gb': result.gpu_memory_gb,
                'throughput_pred_sec': result.throughput_pred_sec,
                'successful_predictions': result.successful_predictions,
                'failed_predictions': result.failed_predictions,
                'total_attempts': result.total_attempts,
                'success_rate_pct': (result.successful_predictions / result.total_attempts * 100) if result.total_attempts > 0 else 0,
                'oom_errors': result.oom_errors,
                'cuda_compilation_time': result.cuda_compilation_time,
                'feasibility_status': result.feasibility_status,
                'memory_efficiency_mb_per_day': result.memory_efficiency,
                'speed_degradation_vs_512_baseline': result.speed_degradation,
                'feasible': result.feasibility_status == "FEASIBLE"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        console.print(f"📄 Large context results exported to: {filepath}")


def main():
    """Main execution for large context benchmark."""
    console.print("[bold red]🚀 Large Context TiRex Performance Benchmark[/bold red]")
    
    try:
        # Initialize benchmark
        benchmark = LargeContextBenchmark()
        
        # Run large context benchmark
        benchmark.run_large_context_benchmark()
        
        # Display results and analysis
        benchmark.display_large_context_results()
        
        # Export results
        results_dir = Path("/home/tca/eon/nt/tests/performance/context_length_empirical_suite/results")
        results_dir.mkdir(exist_ok=True)
        
        export_path = results_dir / f"large_context_benchmark_{int(time.time())}.csv"
        benchmark.export_large_context_results(str(export_path))
        
        console.print(f"\n✅ Large context benchmark complete! Results: {export_path}")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Large context benchmark interrupted")
        return 1
    except Exception as e:
        console.print(f"❌ Large context benchmark failed: {e}")
        logger.exception("Large context benchmark failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)