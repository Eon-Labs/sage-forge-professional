"""
Centralized cache directory configuration using platformdirs.

This module provides standardized cache directory paths following 2024-2025 best practices
for cross-platform Python applications. Uses platformdirs for XDG compliance on Linux
and appropriate platform-specific locations on macOS/Windows.
"""

from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir, user_data_dir
from rich.console import Console

console = Console()

# Application information for platformdirs
APP_NAME = "nautilus-test"
APP_AUTHOR = "nautilus-trader"
APP_VERSION = "0.1.0"


class CacheDirectoryManager:
    """
    Manages cache directories using platformdirs best practices (2024-2025).
    
    Provides centralized cache directory management with automatic creation
    and platform-specific path resolution following XDG Base Directory Specification
    on Linux and appropriate equivalents on other platforms.
    
    Examples:
    - Linux: ~/.cache/nautilus-test/
    - macOS: ~/Library/Caches/nautilus-test/
    - Windows: %LOCALAPPDATA%/nautilus-test/Cache/
    """

    def __init__(self):
        """Initialize cache directory manager with platformdirs."""
        self._base_cache_dir = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
        self._base_data_dir = Path(user_data_dir(APP_NAME, APP_AUTHOR))

        # Create base directories
        self._base_cache_dir.mkdir(parents=True, exist_ok=True)
        self._base_data_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]📁 Cache directory: {self._base_cache_dir}[/cyan]")
        console.print(f"[cyan]📁 Data directory: {self._base_data_dir}[/cyan]")

    @property
    def base_cache_dir(self) -> Path:
        """Get the base cache directory path."""
        return self._base_cache_dir

    @property
    def base_data_dir(self) -> Path:
        """Get the base data directory path."""
        return self._base_data_dir

    def get_cache_dir(self, subdirectory: str) -> Path:
        """
        Get a specific cache subdirectory.
        
        Parameters
        ----------
        subdirectory : str
            Name of the cache subdirectory (e.g., 'funding', 'market_data').
            
        Returns
        -------
        Path
            Platform-specific cache directory path.
        """
        cache_dir = self._base_cache_dir / subdirectory
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def get_data_dir(self, subdirectory: str) -> Path:
        """
        Get a specific data subdirectory.
        
        Parameters
        ----------
        subdirectory : str
            Name of the data subdirectory (e.g., 'historical', 'backtest_results').
            
        Returns
        -------
        Path
            Platform-specific data directory path.
        """
        data_dir = self._base_data_dir / subdirectory
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def clear_cache(self, subdirectory: Optional[str] = None) -> None:
        """
        Clear cache directory contents.
        
        Parameters
        ----------
        subdirectory : str, optional
            Specific subdirectory to clear. If None, clears all cache.
        """
        import shutil

        if subdirectory:
            target_dir = self._base_cache_dir / subdirectory
            if target_dir.exists():
                shutil.rmtree(target_dir)
                console.print(f"[yellow]🗑️ Cleared cache: {target_dir}[/yellow]")
        else:
            if self._base_cache_dir.exists():
                shutil.rmtree(self._base_cache_dir)
                self._base_cache_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[yellow]🗑️ Cleared all cache: {self._base_cache_dir}[/yellow]")

    def get_cache_size(self, subdirectory: Optional[str] = None) -> int:
        """
        Get total size of cache directory in bytes.
        
        Parameters
        ----------
        subdirectory : str, optional
            Specific subdirectory to measure. If None, measures all cache.
            
        Returns
        -------
        int
            Total size in bytes.
        """
        target_dir = self._base_cache_dir
        if subdirectory:
            target_dir = target_dir / subdirectory

        if not target_dir.exists():
            return 0

        total_size = 0
        for file_path in target_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    def format_cache_size(self, subdirectory: Optional[str] = None) -> str:
        """
        Get formatted cache size string.
        
        Parameters
        ----------
        subdirectory : str, optional
            Specific subdirectory to measure. If None, measures all cache.
            
        Returns
        -------
        str
            Formatted size string (e.g., "1.5 MB").
        """
        size_bytes = self.get_cache_size(subdirectory)

        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


# Global cache manager instance
cache_manager = CacheDirectoryManager()

# Convenience functions for common cache directories
def get_funding_cache_dir() -> Path:
    """Get funding rate cache directory."""
    return cache_manager.get_cache_dir("funding")

def get_market_data_cache_dir() -> Path:
    """Get market data cache directory."""
    return cache_manager.get_cache_dir("market_data")

def get_dsm_cache_dir() -> Path:
    """Get DSM (Data Source Manager) cache directory."""
    return cache_manager.get_cache_dir("dsm")

def get_backtest_data_dir() -> Path:
    """Get backtest results data directory."""
    return cache_manager.get_data_dir("backtest_results")

def get_historical_data_dir() -> Path:
    """Get historical data directory."""
    return cache_manager.get_data_dir("historical")


# Legacy compatibility function for existing code
def get_legacy_cache_dir(subdirectory: str = "data_cache") -> Path:
    """
    Get cache directory with legacy compatibility.
    
    This function provides backward compatibility for existing code
    while transitioning to platformdirs standard locations.
    
    Parameters
    ----------
    subdirectory : str
        Cache subdirectory name.
        
    Returns
    -------
    Path
        Platform-specific cache directory path.
    """
    return cache_manager.get_cache_dir(subdirectory)


if __name__ == "__main__":
    # Display cache information
    console.print("[bold green]🗂️ Cache Directory Configuration[/bold green]")
    console.print(f"Application: {APP_NAME} v{APP_VERSION}")
    console.print(f"Author: {APP_AUTHOR}")
    console.print()

    console.print("[bold cyan]📁 Directory Paths:[/bold cyan]")
    console.print(f"Base Cache: {cache_manager.base_cache_dir}")
    console.print(f"Base Data: {cache_manager.base_data_dir}")
    console.print()

    console.print("[bold cyan]🏷️ Specialized Directories:[/bold cyan]")
    console.print(f"Funding Cache: {get_funding_cache_dir()}")
    console.print(f"Market Data: {get_market_data_cache_dir()}")
    console.print(f"DSM Cache: {get_dsm_cache_dir()}")
    console.print(f"Backtest Results: {get_backtest_data_dir()}")
    console.print(f"Historical Data: {get_historical_data_dir()}")
    console.print()

    # Show cache sizes if any exist
    total_size = cache_manager.format_cache_size()
    console.print(f"[bold yellow]💾 Total Cache Size: {total_size}[/bold yellow]")
