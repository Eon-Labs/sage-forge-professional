# Cache Management with Platform-Standard Directories

## Overview

As of 2024-2025, the nautilus-test project has migrated from workspace-based cache directories to platform-standard cache locations using [platformdirs](https://platformdirs.readthedocs.io/). This follows current best practices for Python applications and provides better user experience across different operating systems.

## Platform-Standard Cache Locations

### Current Cache Directories

| Platform | Cache Location | Data Location |
|----------|----------------|---------------|
| **macOS** | `~/Library/Caches/nautilus-test/` | `~/Library/Application Support/nautilus-test/` |
| **Linux** | `~/.cache/nautilus-test/` | `~/.local/share/nautilus-test/` |
| **Windows** | `%LOCALAPPDATA%/nautilus-test/Cache/` | `%LOCALAPPDATA%/nautilus-test/Data/` |

### Specialized Cache Subdirectories

- **Funding Cache**: `{cache_dir}/funding/` - Funding rate data and calculations
- **Market Data**: `{cache_dir}/market_data/` - Historical and real-time market data
- **DSM Cache**: `{cache_dir}/dsm/` - Data Source Manager cache files
- **Backtest Results**: `{data_dir}/backtest_results/` - Backtesting output and analysis
- **Historical Data**: `{data_dir}/historical/` - Long-term historical datasets

## Benefits of Platform-Standard Directories

### ✅ Clean Workspace
- Cache files no longer clutter the git repository
- Workspace stays focused on source code and configuration
- No need for extensive .gitignore rules for cache files

### ✅ Cross-Platform Compatibility
- Follows platform conventions (XDG Base Directory Specification on Linux)
- Integrates with system cleanup tools and backup software
- Respects user preferences for data storage locations

### ✅ Centralized Management
- Single configuration point for all cache directories
- Consistent cache size monitoring and cleanup
- Easy migration between different cache strategies

### ✅ Professional Standards
- Follows 2024-2025 Python packaging best practices
- Compatible with modern application packaging standards
- Proper separation of cache, data, and configuration files

## Usage Examples

### Basic Cache Directory Access

```python
from nautilus_test.utils.cache_config import (
    get_funding_cache_dir,
    get_market_data_cache_dir,
    get_dsm_cache_dir,
)

# Get specialized cache directories
funding_cache = get_funding_cache_dir()
market_cache = get_market_data_cache_dir()
dsm_cache = get_dsm_cache_dir()

# Use in your modules
data_manager = ArrowDataManager(cache_dir=market_cache)
funding_provider = FundingRateProvider(cache_dir=funding_cache)
```

### Advanced Cache Management

```python
from nautilus_test.utils.cache_config import cache_manager

# Get cache information
total_size = cache_manager.format_cache_size()
funding_size = cache_manager.format_cache_size("funding")

# Clear specific cache
cache_manager.clear_cache("market_data")

# Clear all cache
cache_manager.clear_cache()

# Get raw directory paths
base_cache = cache_manager.base_cache_dir
base_data = cache_manager.base_data_dir
```

## Migration from Workspace Cache

### Automatic Migration

The project includes a migration script to help transition from old workspace cache directories:

```bash
# Check what would be migrated (dry run)
uv run python scripts/migrate_cache.py --dry-run

# Perform the migration
uv run python scripts/migrate_cache.py

# Only clean old directories (don't migrate data)
uv run python scripts/migrate_cache.py --clean-only
```

### Manual Migration

If you prefer manual migration:

1. **Backup important data** from old cache directories
2. **Run the new system** to create platform-standard directories
3. **Copy important cache files** to new locations if needed
4. **Delete old workspace cache directories**:
   - `data_cache/`
   - `nautilus_test/data_cache/`
   - `funding_integration/`
   - `production_funding/`
   - `dsm_cache/`

## Configuration and Customization

### Environment Variables

The cache system respects standard environment variables:

- **Linux**: `XDG_CACHE_HOME`, `XDG_DATA_HOME`
- **All platforms**: Custom paths can be set via environment variables

### Custom Cache Directories

For development or testing, you can override default directories:

```python
from nautilus_test.utils.cache_config import CacheDirectoryManager
from pathlib import Path

# Use custom cache location
custom_manager = CacheDirectoryManager()
custom_cache = Path("/custom/cache/location")

# Override specific components
data_manager = ArrowDataManager(cache_dir=custom_cache / "market_data")
```

## Monitoring and Maintenance

### Cache Size Monitoring

```bash
# Display cache information
uv run python scripts/test_cache_config.py

# Or programmatically
python -c "
from nautilus_test.utils.cache_config import cache_manager
print(f'Total cache: {cache_manager.format_cache_size()}')
print(f'Funding: {cache_manager.format_cache_size(\"funding\")}')
print(f'Market data: {cache_manager.format_cache_size(\"market_data\")}')
"
```

### Cache Cleanup

Platform-standard directories integrate with system cleanup tools:

- **macOS**: Managed by system cache cleanup
- **Linux**: Compatible with `bleachbit`, `stacer`, and similar tools
- **Windows**: Integrates with Disk Cleanup and similar utilities

### Manual Cleanup

```python
from nautilus_test.utils.cache_config import cache_manager

# Clear specific cache types
cache_manager.clear_cache("funding")
cache_manager.clear_cache("market_data") 
cache_manager.clear_cache("dsm")

# Clear all cache
cache_manager.clear_cache()
```

## Implementation Details

### Dependencies

- **platformdirs>=4.3.8**: Core platform directory resolution
- **pathlib**: Modern path handling
- **rich**: User-friendly console output

### Code Organization

```
src/nautilus_test/utils/
├── cache_config.py          # Main cache configuration module
└── data_manager.py          # Updated to use platform cache

src/nautilus_test/funding/
├── provider.py              # Updated funding cache
├── backtest_integrator.py   # Updated backtest cache
└── ...

scripts/
├── test_cache_config.py     # Cache system demonstration
└── migrate_cache.py         # Migration from workspace cache
```

### Backward Compatibility

The system maintains backward compatibility:

- Old workspace cache directories are still supported if they exist
- Migration is optional and non-destructive
- Legacy `data_cache` parameter still works in all modules

## Troubleshooting

### Permission Issues

If you encounter permission errors:

```bash
# Check cache directory permissions
ls -la ~/Library/Caches/nautilus-test/  # macOS
ls -la ~/.cache/nautilus-test/           # Linux

# Fix permissions if needed
chmod -R 755 ~/.cache/nautilus-test/     # Linux
```

### Disk Space Issues

Monitor cache size and clean up as needed:

```python
from nautilus_test.utils.cache_config import cache_manager

# Check sizes
print(f"Total cache: {cache_manager.format_cache_size()}")

# Clean if too large
if cache_manager.get_cache_size() > 1024*1024*1024:  # 1GB
    cache_manager.clear_cache("market_data")  # Clean largest cache
```

### Custom Platform Locations

For non-standard setups, override environment variables:

```bash
# Linux custom locations
export XDG_CACHE_HOME=/custom/cache
export XDG_DATA_HOME=/custom/data

# Then run nautilus-test normally
uv run python scripts/test_cache_config.py
```

## Best Practices

### Development

1. **Use the provided cache functions** rather than hardcoded paths
2. **Check cache sizes** regularly during development
3. **Clean cache between major changes** to avoid stale data
4. **Test on multiple platforms** to ensure compatibility

### Production

1. **Monitor cache growth** in production environments
2. **Set up automated cleanup** for long-running systems
3. **Configure backup exclusions** for cache directories
4. **Use environment variables** for custom deployment locations

### Testing

1. **Use temporary directories** for unit tests
2. **Clean cache before integration tests** for reproducibility
3. **Test migration scripts** in staging environments
4. **Verify cross-platform compatibility** on target platforms

## Future Enhancements

- **Cache expiration policies**: Automatic cleanup of old cache files
- **Compression**: Automatic compression of large cache files
- **Remote cache**: Support for shared cache in distributed environments
- **Metrics**: Detailed cache usage analytics and reporting