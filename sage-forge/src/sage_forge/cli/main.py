"""
Main CLI entry point for SAGE-Forge.

Provides professional command-line interface with subcommands for:
- setup: Environment validation and configuration
- create: Template generation for strategies and models
- validate: Quick environment and dependency checks
- info: System information and diagnostics
"""

import click
from rich.console import Console

from sage_forge import __version__
from sage_forge.cli import setup, create, validate

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="SAGE-Forge")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """
    🔥 SAGE-Forge: Self-Adaptive Generative Evaluation Framework
    
    Professional infrastructure for developing adaptive trading strategies
    with NautilusTrader, real market data, and zero trial-and-error setup.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        console.print("🔥 [bold blue]SAGE-Forge CLI[/bold blue] - Verbose mode enabled")


@main.command()
@click.pass_context
def info(ctx):
    """Show system information and environment status."""
    verbose = ctx.obj.get('verbose', False)
    
    console.print("\n[bold blue]🔥 SAGE-Forge System Information[/bold blue]")
    console.print(f"Version: [green]{__version__}[/green]")
    
    # Basic system info
    import sys
    import platform
    console.print(f"Python: [cyan]{sys.version.split()[0]}[/cyan]")
    console.print(f"Platform: [cyan]{platform.system()} {platform.release()}[/cyan]")
    
    # Check key dependencies
    deps_status = []
    critical_deps = [
        'nautilus_trader',
        'finplot', 
        'pyqtgraph',
        'binance',
        'tenacity',
        'rich',
        'pandas',
        'polars'
    ]
    
    for dep in critical_deps:
        try:
            __import__(dep)
            deps_status.append(f"✅ {dep}")
        except ImportError:
            deps_status.append(f"❌ {dep}")
    
    console.print("\n[bold]Dependencies Status:[/bold]")
    for status in deps_status:
        console.print(f"  {status}")
    
    # Check environment validation
    from pathlib import Path
    validated_marker = Path('.sage-validated')
    if validated_marker.exists():
        console.print("\n✅ [green]Environment validated and ready[/green]")
    else:
        console.print("\n⚠️ [yellow]Environment not validated - run 'sage-setup'[/yellow]")
    
    if verbose:
        # Additional verbose information
        import os
        console.print(f"\n[dim]Working Directory: {os.getcwd()}[/dim]")
        console.print(f"[dim]Python Executable: {sys.executable}[/dim]")


# Add subcommands
main.add_command(setup.setup)
main.add_command(create.create)
main.add_command(validate.validate)


if __name__ == '__main__':
    main()