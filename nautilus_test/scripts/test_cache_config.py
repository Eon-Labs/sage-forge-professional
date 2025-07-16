#!/usr/bin/env python3
"""
Test script to demonstrate platformdirs cache directory configuration.

This script shows the new cache directory setup using 2024-2025 best practices
with cross-platform compatibility via platformdirs.
"""

import sys
from pathlib import Path

# Add project source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nautilus_test.utils.cache_config import (
    cache_manager,
    get_backtest_data_dir,
    get_dsm_cache_dir,
    get_funding_cache_dir,
    get_historical_data_dir,
    get_market_data_cache_dir,
)

console = Console()


def main():
    """Display cache directory configuration and create test files."""
    # Display header
    console.print(Panel.fit(
        "[bold blue]🗂️ Platform-Standard Cache Directory Configuration[/bold blue]\n"
        "[cyan]Following 2024-2025 best practices with platformdirs[/cyan]",
        border_style="blue",
    ))

    # Create table showing directories
    table = Table(title="📁 Cache Directory Locations")
    table.add_column("Purpose", style="bold green")
    table.add_column("Directory Path", style="cyan")
    table.add_column("Platform Standard", style="yellow")

    import platform
    os_name = platform.system()

    # Determine platform-specific pattern
    if os_name == "Darwin":
        platform_pattern = "~/Library/Caches/nautilus-test/"
    elif os_name == "Linux":
        platform_pattern = "~/.cache/nautilus-test/"
    elif os_name == "Windows":
        platform_pattern = "%LOCALAPPDATA%/nautilus-test/Cache/"
    else:
        platform_pattern = "Platform-specific"

    # Add directories to table
    directories = [
        ("Base Cache", cache_manager.base_cache_dir, f"{platform_pattern}"),
        ("Base Data", cache_manager.base_data_dir, f"{platform_pattern.replace('Cache', 'Data')}"),
        ("Funding Cache", get_funding_cache_dir(), f"{platform_pattern}funding/"),
        ("Market Data", get_market_data_cache_dir(), f"{platform_pattern}market_data/"),
        ("DSM Cache", get_dsm_cache_dir(), f"{platform_pattern}dsm/"),
        ("Backtest Results", get_backtest_data_dir(), f"{platform_pattern.replace('Cache', 'Data')}backtest_results/"),
        ("Historical Data", get_historical_data_dir(), f"{platform_pattern.replace('Cache', 'Data')}historical/"),
    ]

    for purpose, path, standard in directories:
        table.add_row(purpose, str(path), standard)

    console.print(table)
    console.print()

    # Show system information
    console.print(f"[bold yellow]🖥️ Current System: {os_name}[/bold yellow]")
    console.print(f"[bold yellow]📊 Platform Pattern: {platform_pattern}[/bold yellow]")
    console.print()

    # Test cache operations
    console.print("[bold green]🧪 Testing Cache Operations[/bold green]")

    # Create test files in each cache directory
    test_operations = [
        ("funding", get_funding_cache_dir, "test_funding_rates.json"),
        ("market_data", get_market_data_cache_dir, "test_market_data.parquet"),
        ("dsm", get_dsm_cache_dir, "test_dsm_cache.arrow"),
    ]

    for cache_type, get_dir_func, test_file in test_operations:
        cache_dir = get_dir_func()
        test_path = cache_dir / test_file

        # Write test file
        test_path.write_text(f"Test {cache_type} cache file created by platformdirs")
        console.print(f"[green]✅ Created test file: {test_path}[/green]")

    console.print()

    # Show cache sizes
    console.print("[bold cyan]💾 Cache Sizes[/bold cyan]")
    total_size = cache_manager.format_cache_size()
    console.print(f"Total cache size: {total_size}")

    for cache_type, get_dir_func, _ in test_operations:
        cache_dir = get_dir_func()
        size = cache_manager.format_cache_size(cache_dir.name)
        console.print(f"{cache_type.title()} cache: {size}")

    console.print()

    # Migration notice
    console.print(Panel.fit(
        "[bold yellow]📋 Migration from Workspace Cache[/bold yellow]\n\n"
        "[green]✅ Old workspace cache files will no longer clutter git repositories[/green]\n"
        "[green]✅ Cache follows XDG Base Directory Specification on Linux[/green]\n"
        "[green]✅ Platform-appropriate locations on macOS and Windows[/green]\n"
        "[green]✅ Automatic cleanup and management[/green]\n\n"
        "[cyan]Note: You can safely delete old 'data_cache/' directories in workspace[/cyan]",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()
