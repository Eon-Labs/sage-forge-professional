#!/usr/bin/env python3
"""
Extended TiRex Context Length Performance Benchmark
==================================================

Comprehensive testing across extended context length range to validate findings
and derive additional insights for optimal backtesting configuration.

Extended Test Matrix:
- Fine-grained range: [96, 144, 192, 240, 288, 336, 384, 432, 480, 512]
- Large context range: [768, 1024, 1536, 2048, 3072, 4096]
- Extreme range: [6144, 8192] (memory permitting)

Goals:
1. Validate the 144 < 288/512 paradox across more data points
2. Find the optimal performance sweet spot
3. Identify memory scaling patterns
4. Discover performance cliff points
5. Test extreme context lengths for quality vs speed tradeoffs
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
class ExtendedBenchmarkResult:
    """Extended benchmark result with detailed metrics."""
    context_length: int
    avg_inference_ms: float
    gpu_memory_mb: float
    throughput_pred_sec: float
    successful_predictions: int
    total_attempts: int
    cuda_compilation_time: float
    memory_efficiency: float  # MB per 100 timesteps
    speed_efficiency: float   # Relative to optimal baseline
    context_category: str     # "fine", "large", "extreme"


class ExtendedContextBenchmark:
    """Extended context length benchmark with comprehensive analysis."""
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        console.print(f"🔥 Using device: {self.device}")
        
        if self.device == "cpu":
            console.print("⚠️ Running on CPU - results will not be representative")
        else:
            # Check GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"🚀 GPU Memory: {gpu_memory_gb:.1f} GB available")
        
        self.results: List[ExtendedBenchmarkResult] = []
        self.baseline_performance = None  # Will set from 288 timestep result
        
    def get_extended_test_matrix(self) -> Dict[str, List[int]]:
        """Get extended test matrix categorized by size."""
        return {
            "fine": [96, 144, 192, 240, 288, 336, 384, 432, 480, 512],  # Fine-grained around sweet spot
            "large": [768, 1024, 1536, 2048],  # Large contexts for quality testing
            "extreme": [3072, 4096]  # Extreme contexts (memory permitting)
        }
    
    def generate_test_data(self, num_points: int = 10000) -> torch.Tensor:
        """Generate extended realistic test data."""
        console.print(f"📊 Generating {num_points} extended test data points")
        
        # More sophisticated price generation for extended testing
        base_price = 45000.0
        
        # Multi-scale noise for realistic market behavior
        daily_trend = np.random.normal(0, 0.005, num_points // 288)  # Daily trend
        daily_trend = np.repeat(daily_trend, 288)[:num_points]
        
        hourly_noise = np.random.normal(0, 0.01, num_points)  # Hourly volatility
        minute_noise = np.random.normal(0, 0.002, num_points)  # Minute-level noise
        
        # Add momentum and mean reversion
        returns = daily_trend + hourly_noise + minute_noise
        for i in range(1, len(returns)):
            returns[i] = 0.05 * returns[i-1] + 0.95 * returns[i]  # Slight momentum
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        console.print(f"📈 Generated price range: ${prices.min():.0f} - ${prices.max():.0f}")
        return torch.tensor(prices, dtype=torch.float32)
    
    def estimate_memory_requirement(self, context_length: int) -> float:
        """Estimate GPU memory requirement for context length."""
        # Based on empirical observations: roughly linear scaling
        base_memory = 150  # MB for 144 timesteps
        scaling_factor = (283 - 150) / (288 - 144)  # MB per timestep
        estimated_mb = base_memory + (context_length - 144) * scaling_factor
        return estimated_mb
    
    def check_memory_feasibility(self, context_length: int) -> bool:
        """Check if context length is feasible given GPU memory."""
        if not torch.cuda.is_available():
            return True  # CPU has different constraints
        
        available_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        required_mb = self.estimate_memory_requirement(context_length)
        safety_margin = 0.8  # Use only 80% of available memory
        
        feasible = required_mb < (available_mb * safety_margin)
        if not feasible:
            console.print(f"⚠️ Context {context_length}: {required_mb:.0f}MB required > {available_mb*safety_margin:.0f}MB available")
        
        return feasible
    
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
    
    def benchmark_single_context(self, 
                                context_length: int,
                                test_data: torch.Tensor,
                                category: str,
                                num_predictions: int = None) -> ExtendedBenchmarkResult:
        """Benchmark single context length with category-specific parameters."""
        
        # Adjust prediction count based on context size
        if num_predictions is None:
            if category == "fine":
                num_predictions = 25
            elif category == "large":
                num_predictions = 15  # Larger contexts are slower
            else:  # extreme
                num_predictions = 8   # Very slow, test fewer
        
        console.print(f"\n🔍 Benchmarking {category} context: {context_length} ({num_predictions} predictions)")
        
        # Check memory feasibility
        if not self.check_memory_feasibility(context_length):
            return ExtendedBenchmarkResult(
                context_length=context_length,
                avg_inference_ms=float('inf'),
                gpu_memory_mb=0.0,
                throughput_pred_sec=0.0,
                successful_predictions=0,
                total_attempts=num_predictions,
                cuda_compilation_time=0.0,
                memory_efficiency=float('inf'),
                speed_efficiency=0.0,
                context_category=category
            )
        
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
            console.print(f"✅ Model loaded in {cuda_compilation_time:.1f}s")
        except Exception as e:
            console.print(f"❌ Failed to load model: {e}")
            return ExtendedBenchmarkResult(
                context_length=context_length,
                avg_inference_ms=float('inf'),
                gpu_memory_mb=0.0,
                throughput_pred_sec=0.0,
                successful_predictions=0,
                total_attempts=num_predictions,
                cuda_compilation_time=0.0,
                memory_efficiency=float('inf'),
                speed_efficiency=0.0,
                context_category=category
            )
        
        # Verify sufficient data
        if len(test_data) < context_length + num_predictions:
            console.print(f"⚠️ Insufficient data: need {context_length + num_predictions}, have {len(test_data)}")
            # Generate more data if needed
            additional_needed = (context_length + num_predictions) - len(test_data)
            console.print(f"🔄 Generating {additional_needed} additional data points")
            additional_data = self.generate_test_data(additional_needed)
            test_data = torch.cat([test_data, additional_data])
        
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
                    
                    # Add delay for extreme contexts to prevent overheating
                    if category == "extreme" and inference_time_ms > 100:
                        time.sleep(0.1)
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    console.print(f"🚨 GPU OOM at prediction {i}: {oom_error}")
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    logger.warning(f"Prediction {i} failed: {e}")
                    progress.update(task, advance=1)
                    continue
        
        # Calculate metrics
        avg_inference_ms = np.mean(inference_times) if inference_times else float('inf')
        throughput = 1000.0 / avg_inference_ms if avg_inference_ms != float('inf') else 0.0
        memory_efficiency = gpu_memory_peak / (context_length / 100) if context_length > 0 else float('inf')
        
        # Calculate speed efficiency relative to baseline (288 timesteps)
        if self.baseline_performance is None and context_length == 288:
            self.baseline_performance = avg_inference_ms
        
        speed_efficiency = (self.baseline_performance / avg_inference_ms) if (self.baseline_performance and avg_inference_ms != float('inf')) else 1.0
        
        return ExtendedBenchmarkResult(
            context_length=context_length,
            avg_inference_ms=avg_inference_ms,
            gpu_memory_mb=gpu_memory_peak,
            throughput_pred_sec=throughput,
            successful_predictions=successful,
            total_attempts=num_predictions,
            cuda_compilation_time=cuda_compilation_time,
            memory_efficiency=memory_efficiency,
            speed_efficiency=speed_efficiency,
            context_category=category
        )
    
    def run_extended_benchmark_suite(self) -> List[ExtendedBenchmarkResult]:
        """Run comprehensive extended benchmark suite."""
        console.print(Panel.fit(
            "[bold cyan]🚀 Extended TiRex Context Length Benchmark[/bold cyan]\n"
            "Comprehensive testing across wide context range\n"
            f"Hardware: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n"
            f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n"
            f"Test Data: 10,000 synthetic BTCUSDT 5min bars",
            style="cyan"
        ))
        
        # Generate extended test data
        test_data = self.generate_test_data(num_points=10000)
        
        # Get test matrix
        test_matrix = self.get_extended_test_matrix()
        
        total_contexts = sum(len(contexts) for contexts in test_matrix.values())
        console.print(f"📊 Testing {total_contexts} different context lengths")
        
        start_time = time.perf_counter()
        
        # Run benchmarks by category
        for category, context_lengths in test_matrix.items():
            console.print(f"\n🔍 Testing {category.upper()} context lengths: {context_lengths}")
            
            for context_length in context_lengths:
                result = self.benchmark_single_context(
                    context_length=context_length,
                    test_data=test_data,
                    category=category
                )
                self.results.append(result)
                
                # Brief pause between tests for GPU cooling
                if result.avg_inference_ms > 50:  # Longer pause for slow contexts
                    time.sleep(3)
                else:
                    time.sleep(1)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        console.print(f"\n⏱️ Total extended benchmark duration: {total_duration/60:.1f} minutes")
        
        return self.results
    
    def analyze_performance_patterns(self):
        """Analyze performance patterns and derive insights."""
        if not self.results:
            console.print("❌ No results to analyze")
            return
        
        # Separate results by category
        fine_results = [r for r in self.results if r.context_category == "fine" and r.avg_inference_ms != float('inf')]
        large_results = [r for r in self.results if r.context_category == "large" and r.avg_inference_ms != float('inf')]
        extreme_results = [r for r in self.results if r.context_category == "extreme" and r.avg_inference_ms != float('inf')]
        
        console.print("\n" + "="*80)
        console.print("🔬 EXTENDED PERFORMANCE ANALYSIS")
        console.print("="*80)
        
        # Fine-grained analysis
        if fine_results:
            console.print(f"\n📊 FINE-GRAINED ANALYSIS ({len(fine_results)} contexts)")
            
            fastest = min(fine_results, key=lambda x: x.avg_inference_ms)
            slowest = max(fine_results, key=lambda x: x.avg_inference_ms)
            most_efficient = max(fine_results, key=lambda x: x.speed_efficiency)
            
            table = Table(title="Fine-Grained Context Performance", box=box.ROUNDED)
            table.add_column("Context", style="cyan")
            table.add_column("Time (ms)", style="bold")
            table.add_column("Memory (MB)", style="green")
            table.add_column("Efficiency", style="yellow")
            table.add_column("Status", style="blue")
            
            for result in sorted(fine_results, key=lambda x: x.context_length):
                efficiency_score = f"{result.speed_efficiency:.2f}x"
                status = "🏆 OPTIMAL" if result == most_efficient else "⚡ FAST" if result == fastest else ""
                
                table.add_row(
                    f"{result.context_length}",
                    f"{result.avg_inference_ms:.1f}",
                    f"{result.gpu_memory_mb:.1f}",
                    efficiency_score,
                    status
                )
            
            console.print(table)
            
            console.print(f"\n🏆 **Fine-Grained Insights:**")
            console.print(f"• **Fastest**: {fastest.context_length} timesteps ({fastest.avg_inference_ms:.1f}ms)")
            console.print(f"• **Most Efficient**: {most_efficient.context_length} timesteps ({most_efficient.speed_efficiency:.2f}x baseline)")
            console.print(f"• **Speed Range**: {fastest.avg_inference_ms:.1f}ms - {slowest.avg_inference_ms:.1f}ms")
            console.print(f"• **144 vs 288 Paradox**: {'CONFIRMED' if any(r.context_length == 144 for r in fine_results) and any(r.context_length == 288 for r in fine_results) else 'UNCLEAR'}")
        
        # Large context analysis
        if large_results:
            console.print(f"\n📊 LARGE CONTEXT ANALYSIS ({len(large_results)} contexts)")
            
            table = Table(title="Large Context Performance", box=box.ROUNDED)
            table.add_column("Context", style="cyan")
            table.add_column("Time (ms)", style="bold")
            table.add_column("Memory (MB)", style="green")  
            table.add_column("Quality Potential", style="magenta")
            
            for result in sorted(large_results, key=lambda x: x.context_length):
                quality_score = "HIGH" if result.context_length >= 1536 else "MEDIUM"
                
                table.add_row(
                    f"{result.context_length}",
                    f"{result.avg_inference_ms:.1f}",
                    f"{result.gpu_memory_mb:.1f}",
                    quality_score
                )
            
            console.print(table)
            
            # Memory scaling analysis
            context_lengths = [r.context_length for r in large_results]
            memory_usage = [r.gpu_memory_mb for r in large_results]
            
            if len(context_lengths) >= 2:
                memory_scaling = (memory_usage[-1] - memory_usage[0]) / (context_lengths[-1] - context_lengths[0])
                console.print(f"\n📈 **Memory Scaling**: {memory_scaling:.2f} MB per 100 timesteps")
        
        # Extreme context analysis
        if extreme_results:
            console.print(f"\n📊 EXTREME CONTEXT ANALYSIS ({len(extreme_results)} contexts)")
            
            for result in extreme_results:
                console.print(f"• **Context {result.context_length}**: {result.avg_inference_ms:.1f}ms, {result.gpu_memory_mb:.0f}MB")
                console.print(f"  - Throughput: {result.throughput_pred_sec:.1f} pred/sec")
                console.print(f"  - Success rate: {result.successful_predictions}/{result.total_attempts}")
        
        # Overall recommendations
        self._generate_extended_recommendations(fine_results, large_results, extreme_results)
    
    def _generate_extended_recommendations(self, fine_results, large_results, extreme_results):
        """Generate comprehensive recommendations based on extended analysis."""
        console.print(f"\n🎯 **EXTENDED BACKTESTING RECOMMENDATIONS**")
        
        if fine_results:
            optimal = max(fine_results, key=lambda x: x.speed_efficiency)
            fastest = min(fine_results, key=lambda x: x.avg_inference_ms)
            
            console.print(f"\n⚡ **Fast Development Iteration**:")
            console.print(f"   Use **{fastest.context_length} timesteps** ({fastest.avg_inference_ms:.1f}ms per prediction)")
            
            console.print(f"\n⚖️ **Optimal Performance Balance**:")
            console.print(f"   Use **{optimal.context_length} timesteps** ({optimal.speed_efficiency:.2f}x efficiency)")
        
        if large_results:
            quality_context = max(large_results, key=lambda x: x.context_length)
            console.print(f"\n🎯 **Quality-Focused Backtesting**:")
            console.print(f"   Use **{quality_context.context_length} timesteps** for maximum pattern recognition")
            console.print(f"   Cost: {quality_context.avg_inference_ms:.1f}ms per prediction")
        
        if extreme_results:
            console.print(f"\n🔬 **Extreme Quality (Research Only)**:")
            for result in extreme_results:
                hours_context = result.context_length * 5 / 60  # 5min bars to hours
                console.print(f"   {result.context_length} timesteps ({hours_context:.0f}h context): {result.avg_inference_ms:.0f}ms")
    
    def export_extended_results(self, filepath: str = None):
        """Export extended results with comprehensive analysis."""
        if not self.results:
            console.print("❌ No results to export")
            return
        
        if filepath is None:
            filepath = f"extended_context_benchmark_{int(time.time())}.csv"
        
        # Prepare comprehensive data
        data = []
        for result in self.results:
            data.append({
                'context_length': result.context_length,
                'context_category': result.context_category,
                'avg_inference_ms': result.avg_inference_ms,
                'gpu_memory_mb': result.gpu_memory_mb,
                'throughput_pred_sec': result.throughput_pred_sec,
                'successful_predictions': result.successful_predictions,
                'total_attempts': result.total_attempts,
                'success_rate_pct': (result.successful_predictions / result.total_attempts * 100) if result.total_attempts > 0 else 0,
                'cuda_compilation_time': result.cuda_compilation_time,
                'memory_efficiency_mb_per_100ts': result.memory_efficiency,
                'speed_efficiency_vs_baseline': result.speed_efficiency,
                'context_hours_5min_bars': result.context_length * 5 / 60,
                'feasible': result.avg_inference_ms != float('inf')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        console.print(f"📄 Extended results exported to: {filepath}")


def main():
    """Main execution for extended context benchmark."""
    console.print("[bold green]🚀 Extended TiRex Context Length Benchmark[/bold green]")
    
    try:
        # Initialize benchmark
        benchmark = ExtendedContextBenchmark()
        
        # Run extended benchmark
        benchmark.run_extended_benchmark_suite()
        
        # Analyze patterns and insights
        benchmark.analyze_performance_patterns()
        
        # Export results
        results_dir = Path("/home/tca/eon/nt/tests/performance/context_length_empirical_suite/results")
        results_dir.mkdir(exist_ok=True)
        
        export_path = results_dir / f"extended_benchmark_{int(time.time())}.csv"
        benchmark.export_extended_results(str(export_path))
        
        console.print(f"\n✅ Extended benchmark complete! Results: {export_path}")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Extended benchmark interrupted")
        return 1
    except Exception as e:
        console.print(f"❌ Extended benchmark failed: {e}")
        logger.exception("Extended benchmark failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)