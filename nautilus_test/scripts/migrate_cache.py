#!/usr/bin/env python3
"""
Cache Migration Script: Workspace to Platform-Standard Directories

This script helps migrate from old workspace cache directories to the new
platform-standard cache locations using platformdirs (2024-2025 best practices).

Usage:
    python scripts/migrate_cache.py [--dry-run] [--clean-only]
    
Options:
    --dry-run     Show what would be done without making changes
    --clean-only  Only clean old cache directories, don't migrate data
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Add project source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nautilus_test.utils.cache_config import cache_manager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


def find_workspace_cache_directories() -> List[Path]:
    """Find old workspace cache directories that need migration."""
    workspace_root = Path(__file__).parent.parent.parent
    
    old_cache_dirs = []
    
    # Check common old cache locations
    potential_dirs = [
        workspace_root / "data_cache",
        workspace_root / "nautilus_test" / "data_cache",
        workspace_root / "nautilus_test" / "funding_integration",
        workspace_root / "nautilus_test" / "production_funding",
        workspace_root / "nautilus_test" / "dsm_cache",
        workspace_root / "nautilus_test" / "tmp" / "funding_rate_cache",
        workspace_root / "nautilus_test" / "tmp" / "funding_rate_test",
    ]
    
    for cache_dir in potential_dirs:
        if cache_dir.exists() and cache_dir.is_dir():
            # Check if it contains actual cache files
            cache_files = list(cache_dir.rglob('*'))
            if cache_files:
                old_cache_dirs.append(cache_dir)
    
    return old_cache_dirs


def get_cache_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except (OSError, ValueError):
                # Skip files that can't be accessed
                pass
    return total_size


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def migrate_cache_data(old_dirs: List[Path], dry_run: bool = False) -> bool:
    """
    Migrate useful cache data to new platform directories.
    
    Parameters
    ----------
    old_dirs : List[Path]
        Old cache directories to migrate from.
    dry_run : bool
        If True, show what would be done without making changes.
        
    Returns
    -------
    bool
        True if migration was successful or would be successful.
    """
    console.print("[bold blue]📦 Migrating Cache Data[/bold blue]")
    
    migrations = []
    
    for old_dir in old_dirs:
        # Determine appropriate new location based on old directory name
        if "funding" in old_dir.name.lower():
            new_dir = cache_manager.get_cache_dir("funding")
            data_type = "Funding"
        elif "market" in old_dir.name.lower() or "data" in old_dir.name.lower():
            new_dir = cache_manager.get_cache_dir("market_data")
            data_type = "Market Data"
        elif "dsm" in old_dir.name.lower():
            new_dir = cache_manager.get_cache_dir("dsm")
            data_type = "DSM"
        else:
            new_dir = cache_manager.get_cache_dir("legacy")
            data_type = "Legacy"
        
        migrations.append((old_dir, new_dir, data_type))
    
    if not migrations:
        console.print("[green]✅ No cache data to migrate[/green]")
        return True
    
    # Show migration plan
    table = Table(title="📋 Migration Plan")
    table.add_column("Data Type", style="bold")
    table.add_column("From (Workspace)", style="red")
    table.add_column("To (Platform Standard)", style="green")
    table.add_column("Size", style="cyan")
    
    for old_dir, new_dir, data_type in migrations:
        size = format_size(get_cache_size(old_dir))
        table.add_row(data_type, str(old_dir), str(new_dir), size)
    
    console.print(table)
    
    if dry_run:
        console.print("[yellow]🔍 DRY RUN: No files were actually migrated[/yellow]")
        return True
    
    # Perform migration
    success = True
    for old_dir, new_dir, data_type in migrations:
        try:
            console.print(f"[cyan]📁 Migrating {data_type} data...[/cyan]")
            
            # Copy useful files (skip temporary and lock files)
            for file_path in old_dir.rglob('*'):
                if file_path.is_file():
                    # Skip temporary and lock files
                    if any(skip in file_path.name.lower() for skip in ['.lock', '.tmp', '.temp', 'temp_']):
                        continue
                    
                    # Copy to new location
                    relative_path = file_path.relative_to(old_dir)
                    new_file_path = new_dir / relative_path
                    new_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if not new_file_path.exists():  # Don't overwrite existing files
                        shutil.copy2(file_path, new_file_path)
                        console.print(f"[green]  ✅ Copied: {relative_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to migrate {data_type}: {e}[/red]")
            success = False
    
    return success


def clean_old_cache_directories(old_dirs: List[Path], dry_run: bool = False) -> bool:
    """
    Clean up old workspace cache directories.
    
    Parameters
    ----------
    old_dirs : List[Path]
        Old cache directories to clean.
    dry_run : bool
        If True, show what would be done without making changes.
        
    Returns
    -------
    bool
        True if cleanup was successful or would be successful.
    """
    console.print("[bold yellow]🗑️ Cleaning Old Cache Directories[/bold yellow]")
    
    if not old_dirs:
        console.print("[green]✅ No old cache directories to clean[/green]")
        return True
    
    # Show cleanup plan
    table = Table(title="🗑️ Cleanup Plan")
    table.add_column("Directory", style="red")
    table.add_column("Size", style="cyan")
    table.add_column("Files", style="yellow")
    
    total_size = 0
    total_files = 0
    
    for old_dir in old_dirs:
        size = get_cache_size(old_dir)
        files = len(list(old_dir.rglob('*')))
        total_size += size
        total_files += files
        
        table.add_row(str(old_dir), format_size(size), str(files))
    
    table.add_section()
    table.add_row("[bold]TOTAL", f"[bold]{format_size(total_size)}", f"[bold]{total_files}")
    
    console.print(table)
    
    if dry_run:
        console.print("[yellow]🔍 DRY RUN: No directories were actually deleted[/yellow]")
        return True
    
    # Confirm deletion
    if not Confirm.ask(f"[bold red]Delete {len(old_dirs)} old cache directories ({format_size(total_size)})?[/bold red]"):
        console.print("[yellow]⏸️ Cleanup cancelled by user[/yellow]")
        return False
    
    # Perform cleanup
    success = True
    for old_dir in old_dirs:
        try:
            shutil.rmtree(old_dir)
            console.print(f"[green]✅ Deleted: {old_dir}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to delete {old_dir}: {e}[/red]")
            success = False
    
    return success


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="Migrate cache from workspace to platform-standard directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--clean-only", action="store_true", help="Only clean old cache directories, don't migrate")
    
    args = parser.parse_args()
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]🚀 Cache Migration: Workspace → Platform Standard[/bold blue]\n"
        "[cyan]Moving to platformdirs (2024-2025 best practices)[/cyan]",
        border_style="blue"
    ))
    
    # Show new cache location
    console.print(f"[bold green]📁 New cache location: {cache_manager.base_cache_dir}[/bold green]")
    console.print()
    
    # Find old cache directories
    console.print("[bold cyan]🔍 Scanning for old cache directories...[/bold cyan]")
    old_dirs = find_workspace_cache_directories()
    
    if not old_dirs:
        console.print("[green]✅ No old cache directories found![/green]")
        console.print("[green]Your workspace is already clean.[/green]")
        return
    
    console.print(f"[yellow]📂 Found {len(old_dirs)} old cache directories[/yellow]")
    console.print()
    
    # Migrate or clean
    if args.clean_only:
        success = clean_old_cache_directories(old_dirs, args.dry_run)
    else:
        # First migrate useful data
        migrate_success = migrate_cache_data(old_dirs, args.dry_run)
        console.print()
        
        # Then clean old directories
        clean_success = clean_old_cache_directories(old_dirs, args.dry_run)
        success = migrate_success and clean_success
    
    console.print()
    
    # Final status
    if success:
        if args.dry_run:
            console.print(Panel.fit(
                "[bold green]✅ Migration Plan Validated[/bold green]\n"
                "[cyan]Run without --dry-run to execute the migration[/cyan]",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                "[bold green]🎉 Migration Completed Successfully![/bold green]\n"
                "[cyan]Your cache is now using platform-standard directories[/cyan]\n"
                "[yellow]Old workspace cache directories have been cleaned[/yellow]",
                border_style="green"
            ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ Migration Failed[/bold red]\n"
            "[yellow]Please check the errors above and try again[/yellow]",
            border_style="red"
        ))


if __name__ == "__main__":
    main()