"""
Configuration management for SAGE-Forge.

Provides centralized configuration with:
- Environment-based settings
- Validation and defaults
- Type safety and documentation  
- Integration with external systems
- Warning management for third-party dependencies
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

import platformdirs


def configure_sage_forge_warnings():
    """
    Configure warnings for SAGE-Forge Professional.
    
    Suppresses known third-party deprecation warnings that don't affect
    functionality but clutter test output and logs.
    
    IMPORTANT: Only suppresses third-party warnings. All SAGE-Forge warnings
    should be addressed immediately.
    """
    # Third-party deprecation warnings to suppress
    third_party_warning_patterns = [
        # xLSTM library (used by TiRex) - PyTorch 2.x compatibility
        ".*torch.cuda.amp.custom_fwd.*",
        ".*torch.cuda.amp.custom_bwd.*",
        
        # CUDA compilation warnings
        ".*TORCH_CUDA_ARCH_LIST.*",
        
        # Additional PyTorch extension warnings
        ".*torch.utils.cpp_extension.*",
        ".*cpp_extension.*",
        ".*ninja.*",  # Build system messages
    ]
    
    # Suppress specific third-party warnings
    for pattern in third_party_warning_patterns:
        warnings.filterwarnings('ignore', category=FutureWarning, message=pattern)
        warnings.filterwarnings('ignore', category=UserWarning, message=pattern)
    
    # Note: We deliberately DO NOT suppress all warnings to catch our own issues


@dataclass
class SageConfig:
    """
    Centralized configuration for SAGE-Forge.
    
    Manages all settings for data sources, models, strategies,
    and system configuration with proper defaults and validation.
    """
    
    # Core System Information
    version: str = "0.1.0"
    name: str = "SAGE-Forge" 
    
    # Data Management
    data_dir: Path = field(default_factory=lambda: Path(platformdirs.user_data_dir("sage-forge")))
    cache_dir: Path = field(default_factory=lambda: Path(platformdirs.user_cache_dir("sage-forge")))
    
    # DSM Configuration
    dsm_provider: str = "binance"
    dsm_market_type: str = "futures_usdt"
    dsm_cache_enabled: bool = True
    dsm_data_quality_threshold: float = 0.95  # 95% minimum data quality
    
    # Model Configuration
    model_cache_enabled: bool = True
    model_auto_retrain: bool = True
    model_uncertainty_threshold: float = 0.1
    
    # Strategy Configuration
    strategy_risk_limit: float = 0.02  # 2% position size limit
    strategy_signal_threshold: float = 0.1
    strategy_max_position_hold: int = 240  # bars
    
    # Backtesting Configuration
    backtest_start_balance: float = 10000.0
    backtest_commission: float = 0.0004  # 0.04% commission
    backtest_slippage: float = 0.0001  # 0.01% slippage
    
    # Visualization Configuration
    chart_theme: str = "dark"
    chart_auto_display: bool = True
    chart_save_enabled: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_performance: bool = True
    
    # System Configuration
    max_workers: int = 4
    memory_limit_mb: int = 4096
    gpu_enabled: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment overrides
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'SAGE_FORGE_DATA_DIR': ('data_dir', Path),
            'SAGE_FORGE_CACHE_DIR': ('cache_dir', Path),
            'SAGE_FORGE_DSM_PROVIDER': ('dsm_provider', str),
            'SAGE_FORGE_DSM_MARKET_TYPE': ('dsm_market_type', str),
            'SAGE_FORGE_LOG_LEVEL': ('log_level', str),
            'SAGE_FORGE_RISK_LIMIT': ('strategy_risk_limit', float),
            'SAGE_FORGE_SIGNAL_THRESHOLD': ('strategy_signal_threshold', float),
            'SAGE_FORGE_MAX_WORKERS': ('max_workers', int),
            'SAGE_FORGE_GPU_ENABLED': ('gpu_enabled', bool),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif attr_type == Path:
                        value = Path(env_value)
                    else:
                        value = attr_type(env_value)
                    
                    setattr(self, attr_name, value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {env_var}: {env_value} ({e})")
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate risk limits
        if not 0.001 <= self.strategy_risk_limit <= 0.1:
            raise ValueError("Risk limit must be between 0.1% and 10%")
        
        # Validate signal threshold
        if not 0.01 <= self.strategy_signal_threshold <= 1.0:
            raise ValueError("Signal threshold must be between 1% and 100%")
        
        # Validate data quality threshold
        if not 0.5 <= self.dsm_data_quality_threshold <= 1.0:
            raise ValueError("Data quality threshold must be between 50% and 100%")
        
        # Validate system limits
        if self.max_workers < 1:
            raise ValueError("Max workers must be at least 1")
        
        if self.memory_limit_mb < 512:
            raise ValueError("Memory limit must be at least 512MB")
    
    def get_dsm_config(self) -> Dict[str, Any]:
        """Get DSM-specific configuration."""
        return {
            'provider': self.dsm_provider,
            'market_type': self.dsm_market_type,
            'cache_enabled': self.dsm_cache_enabled,
            'data_quality_threshold': self.dsm_data_quality_threshold,
            'cache_dir': self.cache_dir / 'dsm',
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            'cache_enabled': self.model_cache_enabled,
            'auto_retrain': self.model_auto_retrain,
            'uncertainty_threshold': self.model_uncertainty_threshold,
            'cache_dir': self.cache_dir / 'models',
            'max_workers': self.max_workers,
            'gpu_enabled': self.gpu_enabled,
        }
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy-specific configuration."""
        return {
            'risk_limit': self.strategy_risk_limit,
            'signal_threshold': self.strategy_signal_threshold,
            'max_position_hold': self.strategy_max_position_hold,
            'log_performance': self.log_performance,
        }
    
    def get_funding_config(self) -> Dict[str, Any]:
        """Get funding rate configuration."""
        return {
            'max_funding_rate': 0.01,  # 1% maximum funding rate
            'min_funding_rate': -0.01,  # -1% minimum funding rate
            'alert_threshold': 50.0,    # Alert for payments over $50
            'daily_limit': 500.0,       # Daily funding limit $500
            'max_single_payment': 1000.0,  # Maximum single payment $1000
            'dsm_max_days': 60,         # DSM historical limit
            'api_timeout': 30,          # API timeout seconds
            'cache_enabled': True,      # Enable funding data caching
        }
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return {
            'start_balance': self.backtest_start_balance,
            'commission': self.backtest_commission,
            'slippage': self.backtest_slippage,
            'data_dir': self.data_dir / 'backtest',
            'results_dir': self.data_dir / 'results',
        }
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return {
            'theme': self.chart_theme,
            'auto_display': self.chart_auto_display,
            'save_enabled': self.chart_save_enabled,
            'output_dir': self.data_dir / 'charts',
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': self.log_level,
            'to_file': self.log_to_file,
            'performance': self.log_performance,
            'log_dir': self.data_dir / 'logs',
        }
    
    def save_to_file(self, filepath: Optional[Path] = None) -> Path:
        """
        Save configuration to file.
        
        Parameters:
        -----------
        filepath : Path, optional
            Path to save configuration file
            
        Returns:
        --------
        filepath : Path
            Path where configuration was saved
        """
        if filepath is None:
            filepath = self.data_dir / 'config.yaml'
        
        import yaml
        
        # Convert config to dictionary
        config_dict = {
            'data_management': {
                'data_dir': str(self.data_dir),
                'cache_dir': str(self.cache_dir),
            },
            'dsm': self.get_dsm_config(),
            'models': self.get_model_config(),
            'strategies': self.get_strategy_config(),
            'backtesting': self.get_backtest_config(),
            'visualization': self.get_visualization_config(),
            'logging': self.get_logging_config(),
            'system': {
                'max_workers': self.max_workers,
                'memory_limit_mb': self.memory_limit_mb,
                'gpu_enabled': self.gpu_enabled,
            }
        }
        
        # Save to YAML file
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'SageConfig':
        """
        Load configuration from file.
        
        Parameters:
        -----------
        filepath : Path
            Path to configuration file
            
        Returns:
        --------
        config : SageConfig
            Loaded configuration
        """
        import yaml
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config with overrides
        config = cls()
        
        # Apply loaded settings
        if 'data_management' in config_dict:
            dm = config_dict['data_management']
            if 'data_dir' in dm:
                config.data_dir = Path(dm['data_dir'])
            if 'cache_dir' in dm:
                config.cache_dir = Path(dm['cache_dir'])
        
        if 'dsm' in config_dict:
            dsm = config_dict['dsm']
            config.dsm_provider = dsm.get('provider', config.dsm_provider)
            config.dsm_market_type = dsm.get('market_type', config.dsm_market_type)
            config.dsm_cache_enabled = dsm.get('cache_enabled', config.dsm_cache_enabled)
        
        # Apply other sections similarly...
        
        return config
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"SageConfig(\n"
            f"  data_dir={self.data_dir},\n"
            f"  dsm_provider={self.dsm_provider},\n"
            f"  risk_limit={self.strategy_risk_limit},\n"
            f"  signal_threshold={self.strategy_signal_threshold}\n"
            f")"
        )


# Global configuration instance
_global_config: Optional[SageConfig] = None


def get_config() -> SageConfig:
    """
    Get global SAGE-Forge configuration.
    
    Returns:
    --------
    config : SageConfig
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = SageConfig()
    return _global_config


def set_config(config: SageConfig) -> None:
    """
    Set global SAGE-Forge configuration.
    
    Parameters:
    -----------
    config : SageConfig
        Configuration to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> SageConfig:
    """
    Reset global configuration to defaults.
    
    Returns:
    --------
    config : SageConfig
        New default configuration
    """
    global _global_config
    _global_config = SageConfig()
    return _global_config