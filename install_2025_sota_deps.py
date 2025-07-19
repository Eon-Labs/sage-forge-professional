#!/usr/bin/env python3
"""
🚀 2025 SOTA Dependencies Installer
==================================

Installs the required dependencies for the Enhanced 2025 SOTA Trading Strategy:
- Optuna (hyperparameter optimization)
- SciPy (statistical functions)
- Scikit-learn (machine learning algorithms)

These libraries enable:
- Auto-tuning with Optuna (parameter-free optimization)
- Bayesian regime detection with confidence scoring
- Ensemble signal generation with ML algorithms
- Advanced statistical analysis
"""

import subprocess
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

def install_package(package):
    """Install a package using pip."""
    try:
        console.print(f"[cyan]📦 Installing {package}...[/cyan]")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]✅ {package} installed successfully[/green]")
            return True
        else:
            console.print(f"[red]❌ Failed to install {package}: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error installing {package}: {e}[/red]")
        return False

def main():
    """Install all 2025 SOTA dependencies."""
    console.print(Panel.fit(
        "[bold cyan]🚀 2025 SOTA Trading Strategy Dependencies Installer[/bold cyan]\n"
        "Installing advanced libraries for state-of-the-art algorithmic trading",
        title="2025 SOTA INSTALLER",
    ))
    
    # List of required packages
    packages = [
        "optuna>=3.0.0",      # Hyperparameter optimization framework
        "scipy>=1.10.0",      # Scientific computing library  
        "scikit-learn>=1.3.0" # Machine learning algorithms
    ]
    
    console.print("\n[bold yellow]📋 Installing 2025 SOTA Dependencies:[/bold yellow]")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
        console.print()  # Empty line for readability
    
    # Installation summary
    console.print("="*60)
    if success_count == len(packages):
        console.print("[bold green]🎉 All dependencies installed successfully![/bold green]")
        console.print("\n[cyan]📊 Enhanced 2025 SOTA features now available:[/cyan]")
        console.print("[cyan]  • Auto-tuning with Optuna (parameter-free optimization)[/cyan]")
        console.print("[cyan]  • Bayesian regime detection with confidence scoring[/cyan]") 
        console.print("[cyan]  • Ensemble signal generation with ML algorithms[/cyan]")
        console.print("[cyan]  • Kelly criterion position sizing with drawdown protection[/cyan]")
        console.print("[cyan]  • Advanced statistical analysis and optimization[/cyan]")
        
        console.print(f"\n[bold green]✅ Run your enhanced trading strategies to see the improvements![/bold green]")
    else:
        failed_count = len(packages) - success_count
        console.print(f"[bold yellow]⚠️ {success_count}/{len(packages)} packages installed successfully[/bold yellow]")
        console.print(f"[yellow]{failed_count} packages failed to install[/yellow]")
        console.print("\n[yellow]The strategy will fall back to basic methods for missing dependencies.[/yellow]")
    
    console.print("\n" + "="*60)

if __name__ == "__main__":
    main()