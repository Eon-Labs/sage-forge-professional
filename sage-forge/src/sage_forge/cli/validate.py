"""
Quick validation and environment checking command.

Provides fast validation checks for daily development workflow:
- Quick dependency verification
- Environment status check
- Basic functionality tests
- Performance benchmarks
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class QuickValidator:
    """Fast validation for daily development workflow."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
    
    def run_quick_validation(self) -> bool:
        """Run quick validation checks (5 seconds)."""
        console.print(Panel(
            "[bold blue]⚡ Quick Environment Validation[/bold blue]\n"
            "Fast checks for daily development workflow",
            border_style="blue"
        ))
        
        checks = [
            ("Environment", self._check_environment),
            ("Dependencies", self._check_critical_deps),
            ("Package", self._check_package_import),
            ("CLI Tools", self._check_cli_tools),
        ]
        
        all_passed = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running quick validation...", total=len(checks))
            
            for name, check_func in checks:
                progress.update(task, description=f"Checking {name}...")
                
                try:
                    start_time = time.time()
                    result = check_func()
                    duration = time.time() - start_time
                    
                    self.results[name] = {
                        'status': 'PASS',
                        'duration': duration,
                        'details': result
                    }
                    
                    if self.verbose:
                        console.print(f"  ✅ {name} ({duration:.2f}s)")
                    
                except Exception as e:
                    self.results[name] = {
                        'status': 'FAIL', 
                        'error': str(e),
                        'duration': time.time() - start_time
                    }
                    all_passed = False
                    if self.verbose:
                        console.print(f"  ❌ {name}: {e}")
                
                progress.update(task, advance=1)
        
        self._show_validation_results(all_passed)
        return all_passed
    
    def run_full_validation(self) -> bool:
        """Run comprehensive validation (30 seconds)."""
        console.print(Panel(
            "[bold blue]🔍 Full Environment Validation[/bold blue]\n"
            "Comprehensive validation of all components",
            border_style="blue"
        ))
        
        checks = [
            ("Python Version", self._check_python_version),
            ("UV Package Manager", self._check_uv_available),
            ("Environment Status", self._check_environment),
            ("All Dependencies", self._check_all_dependencies),
            ("Package Import", self._check_package_import),
            ("CLI Tools", self._check_cli_tools),
            ("File Syntax", self._check_syntax),
            ("Import Tests", self._check_imports),
            ("Performance", self._run_performance_test),
        ]
        
        all_passed = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running full validation...", total=len(checks))
            
            for name, check_func in checks:
                progress.update(task, description=f"Validating {name}...")
                
                try:
                    start_time = time.time()
                    result = check_func()
                    duration = time.time() - start_time
                    
                    self.results[name] = {
                        'status': 'PASS',
                        'duration': duration,
                        'details': result
                    }
                    
                    if self.verbose:
                        console.print(f"  ✅ {name} ({duration:.2f}s)")
                    
                except Exception as e:
                    self.results[name] = {
                        'status': 'FAIL',
                        'error': str(e),
                        'duration': time.time() - start_time
                    }
                    all_passed = False
                    console.print(f"  ❌ {name}: {e}")
                
                progress.update(task, advance=1)
        
        self._show_validation_results(all_passed)
        self._show_performance_summary()
        return all_passed
    
    def _check_environment(self) -> Dict[str, Any]:
        """Check environment status."""
        status = {}
        
        # Check if validated marker exists
        validated_marker = Path('.sage-validated')
        status['validated'] = validated_marker.exists()
        
        # Check working directory
        status['in_sage_forge'] = Path('pyproject.toml').exists()
        
        # Check virtual environment
        status['venv_active'] = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        return status
    
    def _check_critical_deps(self) -> List[str]:
        """Check critical dependencies for quick validation."""
        critical_deps = [
            'nautilus_trader',
            'rich',
            'click',
            'pandas',
            'numpy'
        ]
        
        available = []
        for dep in critical_deps:
            try:
                __import__(dep)
                available.append(dep)
            except ImportError:
                pass
        
        if len(available) < len(critical_deps):
            missing = set(critical_deps) - set(available)
            raise RuntimeError(f"Missing critical dependencies: {', '.join(missing)}")
        
        return available
    
    def _check_all_dependencies(self) -> List[str]:
        """Check all project dependencies."""
        all_deps = [
            'nautilus_trader', 'finplot', 'pyqtgraph', 'binance',
            'tenacity', 'rich', 'click', 'pandas', 'polars', 'pyarrow',
            'scikit_learn', 'pycatch22', 'scipy', 'stumpy', 'httpx',
            'loguru', 'platformdirs', 'attrs', 'pendulum'
        ]
        
        available = []
        missing = []
        
        for dep in all_deps:
            try:
                __import__(dep)
                available.append(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")
        
        return available
    
    def _check_package_import(self) -> Dict[str, bool]:
        """Check sage_forge package imports."""
        imports_to_test = [
            'sage_forge',
            'sage_forge.cli',
            'sage_forge.core',
            'sage_forge.models',
            'sage_forge.strategies',
            'sage_forge.data',
            'sage_forge.visualization',
            'sage_forge.utils'
        ]
        
        results = {}
        for import_path in imports_to_test:
            try:
                __import__(import_path)
                results[import_path] = True
            except ImportError:
                results[import_path] = False
        
        # Check if core package imports work
        core_failed = not results.get('sage_forge', False)
        if core_failed:
            raise RuntimeError("Core sage_forge package import failed")
        
        return results
    
    def _check_cli_tools(self) -> List[str]:
        """Check CLI tool availability."""
        tools = ['sage-forge', 'sage-setup', 'sage-create', 'sage-validate']
        available = []
        
        # For now, just check if we're in the right environment
        # In a full installation, these would be available as console scripts
        if Path('src/sage_forge/cli').exists():
            available = tools
        
        return available
    
    def _check_python_version(self) -> str:
        """Check Python version."""
        version = sys.version_info
        if version.major != 3 or version.minor < 12:
            raise RuntimeError(f"Python 3.12+ required, found {version.major}.{version.minor}")
        
        return f"{version.major}.{version.minor}.{version.micro}"
    
    def _check_uv_available(self) -> str:
        """Check UV package manager."""
        import subprocess
        try:
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("UV package manager not found")
    
    def _check_syntax(self) -> int:
        """Check Python file syntax."""
        import py_compile
        
        python_files = list(Path('src').rglob('*.py')) if Path('src').exists() else []
        
        for py_file in python_files:
            try:
                py_compile.compile(py_file, doraise=True)
            except py_compile.PyCompileError as e:
                raise RuntimeError(f"Syntax error in {py_file}: {e}")
        
        return len(python_files)
    
    def _check_imports(self) -> int:
        """Check import functionality."""
        # Test basic imports that should work
        test_imports = [
            'numpy',
            'pandas', 
            'rich.console',
            'click'
        ]
        
        for import_path in test_imports:
            try:
                __import__(import_path)
            except ImportError as e:
                raise RuntimeError(f"Import test failed for {import_path}: {e}")
        
        return len(test_imports)
    
    def _run_performance_test(self) -> Dict[str, float]:
        """Run basic performance benchmark."""
        import numpy as np
        
        # Simple numpy operations benchmark
        start = time.time()
        data = np.random.randn(10000, 100)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        numpy_time = time.time() - start
        
        # Pandas operations benchmark
        start = time.time()
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df_mean = df.mean()
            df_std = df.std()
            pandas_time = time.time() - start
        except ImportError:
            pandas_time = 0.0
        
        return {
            'numpy_ops': numpy_time,
            'pandas_ops': pandas_time,
            'total': numpy_time + pandas_time
        }
    
    def _show_validation_results(self, all_passed: bool):
        """Show validation results table."""
        table = Table(title="🔍 Validation Results", border_style="green" if all_passed else "red")
        table.add_column("Component", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Time", justify="right", style="dim")
        table.add_column("Details", style="dim")
        
        for name, result in self.results.items():
            if result['status'] == 'PASS':
                status = "✅ PASS"
                details = str(result.get('details', ''))[:50]
            else:
                status = "❌ FAIL"
                details = result.get('error', '')[:50]
            
            duration = f"{result.get('duration', 0):.2f}s"
            table.add_row(name, status, duration, details)
        
        console.print(table)
        
        if all_passed:
            console.print("\n🎉 [bold green]All validation checks passed![/bold green]")
        else:
            console.print("\n💥 [bold red]Some validation checks failed![/bold red]")
            console.print("   Run with --verbose for detailed error information")
    
    def _show_performance_summary(self):
        """Show performance benchmark summary."""
        if 'Performance' in self.results and self.results['Performance']['status'] == 'PASS':
            perf = self.results['Performance']['details']
            
            console.print("\n[bold blue]⚡ Performance Summary:[/bold blue]")
            console.print(f"  NumPy operations: {perf['numpy_ops']:.3f}s")
            console.print(f"  Pandas operations: {perf['pandas_ops']:.3f}s")
            console.print(f"  Total benchmark: {perf['total']:.3f}s")
            
            # Performance assessment
            if perf['total'] < 0.1:
                console.print("  [green]⚡ Excellent performance[/green]")
            elif perf['total'] < 0.5:
                console.print("  [yellow]🔄 Good performance[/yellow]")
            else:
                console.print("  [red]🐌 Performance may be degraded[/red]")


@click.command()
@click.option('--quick', '-q', is_flag=True, help='Run quick validation (5 seconds)')
@click.option('--full', '-f', is_flag=True, help='Run full validation (30 seconds)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed validation results')
@click.pass_context
def validate(ctx, quick, full, verbose):
    """
    Quick validation and environment checking.
    
    Provides fast validation checks for daily development workflow.
    Default mode runs quick validation unless --full is specified.
    """
    # Use verbose from parent context if not specified
    if not verbose and ctx.obj:
        verbose = ctx.obj.get('verbose', False)
    
    # Determine validation mode
    if full:
        mode = "full"
    elif quick:
        mode = "quick"
    else:
        # Default to quick validation
        mode = "quick"
    
    validator = QuickValidator(verbose=verbose)
    
    try:
        if mode == "full":
            success = validator.run_full_validation()
        else:
            success = validator.run_quick_validation()
        
        if success:
            console.print("\n✅ [bold green]Environment is ready for development![/bold green]")
            if mode == "quick":
                console.print("   Use --full for comprehensive validation")
        else:
            console.print("\n⚠️ [bold yellow]Some issues found - check results above[/bold yellow]")
            console.print("   Run 'sage-setup' to fix environment issues")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"\n💥 [bold red]Validation failed:[/bold red] {e}")
        sys.exit(1)


def main():
    """Main entry point for sage-validate command."""
    validate()