"""
tsfresh Feature Extractor Wrapper

Integrates automated time series feature extraction and selection.
Generates 1200+ features with statistical relevance testing.

Reference: Christ et al. (2018) "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests"
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

class TSFreshWrapper:
    """
    Wrapper for tsfresh automated feature extraction and selection.
    
    Provides comprehensive feature engineering with built-in statistical selection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'default_fc_parameters': 'comprehensive',
            'n_jobs': 1,  # Conservative for backtesting
            'disable_progressbar': True,
            'fdr_level': 0.05
        }
        self.is_initialized = False
        self.tsfresh_available = False
        
        console.print("[cyan]⚙️ tsfresh wrapper initialized[/cyan]")
    
    def initialize_model(self) -> bool:
        """Initialize tsfresh with dependency checks."""
        try:
            import tsfresh
            from tsfresh import extract_features, select_features
            from tsfresh.utilities.dataframe_functions import impute
            from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
            
            self.tsfresh = tsfresh
            self.tsfresh_extract_features = extract_features
            self.tsfresh_select_features = select_features
            self.impute = impute
            self.ComprehensiveFCParameters = ComprehensiveFCParameters
            self.MinimalFCParameters = MinimalFCParameters
            
            self.tsfresh_available = True
            console.print("[green]✅ tsfresh imported successfully[/green]")
            
        except ImportError as e:
            console.print(f"[yellow]⚠️ tsfresh not available: {e}[/yellow]")
            console.print("[yellow]💡 Using synthetic tsfresh-like features[/yellow]")
            self.tsfresh_available = False
        
        self.is_initialized = True
        return True
    
    def extract_features(self, 
                        time_series_data: Union[pd.DataFrame, pd.Series],
                        column_id: Optional[str] = None,
                        column_sort: Optional[str] = None,
                        column_value: Optional[str] = None,
                        feature_set: str = 'comprehensive') -> pd.DataFrame:
        """
        Extract tsfresh features from time series data.
        
        Args:
            time_series_data: Input time series (DataFrame or Series)
            column_id: ID column name (for DataFrame input)
            column_sort: Time column name (for DataFrame input)  
            column_value: Value column name (for DataFrame input)
            feature_set: 'comprehensive', 'minimal', or 'efficient'
            
        Returns:
            DataFrame with extracted features
        """
        if not self.is_initialized:
            self.initialize_model()
        
        try:
            if self.tsfresh_available:
                return self._extract_real_tsfresh(
                    time_series_data, column_id, column_sort, column_value, feature_set
                )
            else:
                return self._extract_synthetic_tsfresh(time_series_data, feature_set)
                
        except Exception as e:
            console.print(f"[yellow]⚠️ tsfresh extraction failed: {e}, using fallback[/yellow]")
            return self._extract_synthetic_tsfresh(time_series_data, feature_set)
    
    def _extract_real_tsfresh(self, 
                             data: Union[pd.DataFrame, pd.Series],
                             column_id: Optional[str],
                             column_sort: Optional[str], 
                             column_value: Optional[str],
                             feature_set: str) -> pd.DataFrame:
        """Extract features using real tsfresh."""
        console.print(f"[green]🎯 Extracting real tsfresh features (set: {feature_set})[/green]")
        
        # Prepare data in tsfresh format
        if isinstance(data, pd.Series):
            # Convert Series to tsfresh DataFrame format
            tsfresh_data = pd.DataFrame({
                'id': [1] * len(data),
                'time': range(len(data)),
                'value': data.values
            })
            column_id = 'id'
            column_sort = 'time'
            column_value = 'value'
        else:
            tsfresh_data = data.copy()
            if column_id is None:
                tsfresh_data['id'] = 1
                column_id = 'id'
            if column_sort is None and 'time' not in tsfresh_data.columns:
                tsfresh_data['time'] = range(len(tsfresh_data))
                column_sort = 'time'
            if column_value is None:
                column_value = tsfresh_data.columns[0]  # Use first numeric column
        
        # Set feature extraction parameters
        if feature_set == 'comprehensive':
            fc_parameters = self.ComprehensiveFCParameters()
        elif feature_set == 'minimal':
            fc_parameters = self.MinimalFCParameters()
        else:  # efficient
            fc_parameters = self._get_efficient_parameters()
        
        try:
            # Extract features
            console.print(f"[cyan]🔧 Extracting features from {len(tsfresh_data)} data points[/cyan]")
            
            extracted_features = self.tsfresh_extract_features(
                tsfresh_data,
                column_id=column_id,
                column_sort=column_sort,
                column_value=column_value,
                default_fc_parameters=fc_parameters,
                n_jobs=self.config.get('n_jobs', 1),
                disable_progressbar=self.config.get('disable_progressbar', True)
            )
            
            # Handle NaN values
            console.print("[cyan]🔧 Imputing missing values[/cyan]")
            imputed_features = self.impute(extracted_features)
            
            console.print(f"[green]✅ Extracted {imputed_features.shape[1]} real tsfresh features[/green]")
            return imputed_features
            
        except Exception as e:
            console.print(f"[red]❌ Real tsfresh extraction failed: {e}[/red]")
            raise
    
    def _extract_synthetic_tsfresh(self, 
                                  data: Union[pd.DataFrame, pd.Series],
                                  feature_set: str) -> pd.DataFrame:
        """Extract synthetic features mimicking tsfresh patterns."""
        console.print(f"[yellow]🔧 Extracting synthetic tsfresh-like features (set: {feature_set})[/yellow]")
        
        # Convert to Series if DataFrame
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                series = data['close']
            else:
                series = data.iloc[:, 0]  # Use first column
        else:
            series = data
        
        features = {}
        
        # Determine feature count based on set
        if feature_set == 'comprehensive':
            feature_categories = self._get_comprehensive_synthetic_features()
        elif feature_set == 'minimal':
            feature_categories = self._get_minimal_synthetic_features()
        else:  # efficient
            feature_categories = self._get_efficient_synthetic_features()
        
        # Calculate features by category
        for category, calculators in feature_categories.items():
            console.print(f"[cyan]📊 Calculating {category} features[/cyan]")
            
            for feature_name, calculator in calculators.items():
                try:
                    features[f"{category}__{feature_name}"] = calculator(series)
                except Exception as e:
                    console.print(f"[yellow]⚠️ Feature {feature_name} failed: {e}[/yellow]")
                    features[f"{category}__{feature_name}"] = 0.0
        
        # Create DataFrame with single row (aggregate features)
        feature_df = pd.DataFrame([features])
        
        console.print(f"[green]✅ Generated {len(features)} synthetic tsfresh-like features[/green]")
        return feature_df
    
    def _get_comprehensive_synthetic_features(self) -> Dict:
        """Get comprehensive set of synthetic features."""
        return {
            'statistical': {
                'mean': lambda x: float(np.mean(x)),
                'std': lambda x: float(np.std(x)),
                'var': lambda x: float(np.var(x)),
                'min': lambda x: float(np.min(x)),
                'max': lambda x: float(np.max(x)),
                'median': lambda x: float(np.median(x)),
                'skewness': lambda x: float(self._safe_skew(x)),
                'kurtosis': lambda x: float(self._safe_kurtosis(x)),
                'quantile_25': lambda x: float(np.quantile(x, 0.25)),
                'quantile_75': lambda x: float(np.quantile(x, 0.75)),
                'iqr': lambda x: float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
                'range': lambda x: float(np.max(x) - np.min(x)),
                'mean_abs_change': lambda x: float(np.mean(np.abs(np.diff(x)))),
                'mean_change': lambda x: float(np.mean(np.diff(x))),
                'variance_larger_than_std': lambda x: float(np.var(x) > np.std(x))
            },
            'autocorrelation': {
                'autocorr_lag_1': lambda x: float(self._safe_autocorr(x, 1)),
                'autocorr_lag_2': lambda x: float(self._safe_autocorr(x, 2)),
                'autocorr_lag_3': lambda x: float(self._safe_autocorr(x, 3)),
                'partial_autocorr_lag_1': lambda x: float(self._safe_partial_autocorr(x, 1)),
                'partial_autocorr_lag_2': lambda x: float(self._safe_partial_autocorr(x, 2)),
            },
            'frequency': {
                'fft_coefficient_real_0': lambda x: float(np.real(np.fft.fft(x)[0])),
                'fft_coefficient_real_1': lambda x: float(np.real(np.fft.fft(x)[1])) if len(x) > 1 else 0.0,
                'fft_coefficient_imag_0': lambda x: float(np.imag(np.fft.fft(x)[0])),
                'fft_coefficient_imag_1': lambda x: float(np.imag(np.fft.fft(x)[1])) if len(x) > 1 else 0.0,
                'spectral_centroid': lambda x: float(self._spectral_centroid(x)),
                'spectral_rolloff': lambda x: float(self._spectral_rolloff(x)),
                'spectral_entropy': lambda x: float(self._spectral_entropy(x)),
            },
            'distribution': {
                'count_above_mean': lambda x: float(np.sum(x > np.mean(x))),
                'count_below_mean': lambda x: float(np.sum(x < np.mean(x))),
                'ratio_beyond_r_sigma': lambda x: float(self._ratio_beyond_r_sigma(x, 1)),
                'ratio_beyond_r_sigma_2': lambda x: float(self._ratio_beyond_r_sigma(x, 2)),
                'ratio_beyond_r_sigma_3': lambda x: float(self._ratio_beyond_r_sigma(x, 3)),
                'has_duplicate': lambda x: float(len(x) != len(set(x))),
                'has_duplicate_max': lambda x: float(np.sum(x == np.max(x)) > 1),
                'has_duplicate_min': lambda x: float(np.sum(x == np.min(x)) > 1),
            },
            'linear_trend': {
                'linear_trend_slope': lambda x: float(self._linear_trend_slope(x)),
                'linear_trend_intercept': lambda x: float(self._linear_trend_intercept(x)),
                'linear_trend_r_value': lambda x: float(self._linear_trend_r_value(x)),
                'linear_trend_p_value': lambda x: float(self._linear_trend_p_value(x)),
                'linear_trend_stderr': lambda x: float(self._linear_trend_stderr(x)),
            },
            'peaks': {
                'number_peaks_n_1': lambda x: float(self._number_peaks(x, 1)),
                'number_peaks_n_5': lambda x: float(self._number_peaks(x, 5)),
                'number_peaks_n_10': lambda x: float(self._number_peaks(x, 10)),
            },
            'energy': {
                'abs_energy': lambda x: float(np.sum(x ** 2)),
                'sum_values': lambda x: float(np.sum(x)),
                'abs_sum_changes': lambda x: float(np.sum(np.abs(np.diff(x)))),
            }
        }
    
    def _get_minimal_synthetic_features(self) -> Dict:
        """Get minimal set of synthetic features."""
        comprehensive = self._get_comprehensive_synthetic_features()
        return {
            'statistical': {k: v for k, v in list(comprehensive['statistical'].items())[:10]},
            'autocorrelation': {k: v for k, v in list(comprehensive['autocorrelation'].items())[:3]},
            'frequency': {k: v for k, v in list(comprehensive['frequency'].items())[:3]},
        }
    
    def _get_efficient_synthetic_features(self) -> Dict:
        """Get efficient set of synthetic features (balanced)."""
        comprehensive = self._get_comprehensive_synthetic_features()
        return {
            'statistical': {k: v for k, v in list(comprehensive['statistical'].items())[:8]},
            'autocorrelation': {k: v for k, v in list(comprehensive['autocorrelation'].items())[:2]},
            'frequency': {k: v for k, v in list(comprehensive['frequency'].items())[:4]},
            'distribution': {k: v for k, v in list(comprehensive['distribution'].items())[:4]},
            'linear_trend': {k: v for k, v in list(comprehensive['linear_trend'].items())[:3]},
        }
    
    def _get_efficient_parameters(self):
        """Get efficient tsfresh parameters."""
        # This would return a subset of ComprehensiveFCParameters for efficiency
        # For now, return minimal parameters
        return self.MinimalFCParameters()
    
    # Helper functions for synthetic feature calculation
    def _safe_skew(self, x: np.ndarray) -> float:
        """Safe skewness calculation."""
        try:
            from scipy.stats import skew
            return skew(x)
        except:
            return 0.0
    
    def _safe_kurtosis(self, x: np.ndarray) -> float:
        """Safe kurtosis calculation."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(x)
        except:
            return 0.0
    
    def _safe_autocorr(self, x: np.ndarray, lag: int) -> float:
        """Safe autocorrelation calculation."""
        try:
            if len(x) <= lag:
                return 0.0
            x_shifted = np.roll(x, lag)
            return np.corrcoef(x[lag:], x_shifted[lag:])[0, 1]
        except:
            return 0.0
    
    def _safe_partial_autocorr(self, x: np.ndarray, lag: int) -> float:
        """Safe partial autocorrelation calculation."""
        try:
            # Simplified partial autocorrelation approximation
            if len(x) <= lag + 1:
                return 0.0
            return self._safe_autocorr(x, lag) - self._safe_autocorr(x, 1) * self._safe_autocorr(x, lag-1)
        except:
            return 0.0
    
    def _spectral_centroid(self, x: np.ndarray) -> float:
        """Calculate spectral centroid."""
        try:
            fft_vals = np.abs(np.fft.fft(x))
            freqs = np.fft.fftfreq(len(x))
            return np.sum(freqs[:len(freqs)//2] * fft_vals[:len(fft_vals)//2]) / np.sum(fft_vals[:len(fft_vals)//2])
        except:
            return 0.0
    
    def _spectral_rolloff(self, x: np.ndarray) -> float:
        """Calculate spectral rolloff."""
        try:
            fft_vals = np.abs(np.fft.fft(x))
            cumsum = np.cumsum(fft_vals[:len(fft_vals)//2])
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
            return float(rolloff_idx[0] if len(rolloff_idx) > 0 else len(fft_vals)//2)
        except:
            return 0.0
    
    def _spectral_entropy(self, x: np.ndarray) -> float:
        """Calculate spectral entropy."""
        try:
            fft_vals = np.abs(np.fft.fft(x))
            psd = fft_vals ** 2
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]
            return -np.sum(psd_norm * np.log2(psd_norm))
        except:
            return 0.0
    
    def _ratio_beyond_r_sigma(self, x: np.ndarray, r: float) -> float:
        """Calculate ratio of values beyond r standard deviations."""
        try:
            mean_x = np.mean(x)
            std_x = np.std(x)
            return np.sum(np.abs(x - mean_x) > r * std_x) / len(x)
        except:
            return 0.0
    
    def _linear_trend_slope(self, x: np.ndarray) -> float:
        """Calculate linear trend slope."""
        try:
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(range(len(x)), x)
            return slope
        except:
            return 0.0
    
    def _linear_trend_intercept(self, x: np.ndarray) -> float:
        """Calculate linear trend intercept."""
        try:
            from scipy.stats import linregress
            _, intercept, _, _, _ = linregress(range(len(x)), x)
            return intercept
        except:
            return 0.0
    
    def _linear_trend_r_value(self, x: np.ndarray) -> float:
        """Calculate linear trend correlation coefficient."""
        try:
            from scipy.stats import linregress
            _, _, r_value, _, _ = linregress(range(len(x)), x)
            return r_value
        except:
            return 0.0
    
    def _linear_trend_p_value(self, x: np.ndarray) -> float:
        """Calculate linear trend p-value."""
        try:
            from scipy.stats import linregress
            _, _, _, p_value, _ = linregress(range(len(x)), x)
            return p_value
        except:
            return 1.0
    
    def _linear_trend_stderr(self, x: np.ndarray) -> float:
        """Calculate linear trend standard error."""
        try:
            from scipy.stats import linregress
            _, _, _, _, stderr = linregress(range(len(x)), x)
            return stderr
        except:
            return 0.0
    
    def _number_peaks(self, x: np.ndarray, n: int) -> float:
        """Count number of peaks."""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(x, height=np.mean(x) + n * np.std(x))
            return len(peaks)
        except:
            # Fallback peak detection
            peaks = 0
            threshold = np.mean(x) + n * np.std(x)
            for i in range(1, len(x) - 1):
                if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > threshold:
                    peaks += 1
            return float(peaks)
    
    def feature_selection(self, 
                         features: pd.DataFrame,
                         target: pd.Series,
                         ml_task: str = 'regression') -> pd.DataFrame:
        """
        Perform feature selection using statistical tests.
        
        Args:
            features: Extracted features DataFrame
            target: Target variable for selection
            ml_task: 'regression' or 'classification'
            
        Returns:
            DataFrame with selected features
        """
        if not self.tsfresh_available:
            console.print("[yellow]⚠️ tsfresh not available, skipping statistical selection[/yellow]")
            return features
        
        try:
            console.print(f"[cyan]🎯 Performing feature selection ({ml_task})[/cyan]")
            console.print(f"[cyan]📊 Input: {features.shape[1]} features[/cyan]")
            
            selected_features = self.tsfresh_select_features(
                features,
                target,
                ml_task=ml_task,
                fdr_level=self.config.get('fdr_level', 0.05),
                n_jobs=self.config.get('n_jobs', 1)
            )
            
            console.print(f"[green]✅ Selected {selected_features.shape[1]} features (from {features.shape[1]})[/green]")
            return selected_features
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Feature selection failed: {e}[/yellow]")
            return features
    
    def extract_and_select_features(self,
                                   time_series_data: Union[pd.DataFrame, pd.Series],
                                   target: Optional[pd.Series] = None,
                                   **kwargs) -> pd.DataFrame:
        """
        Extract features and optionally perform selection.
        
        Args:
            time_series_data: Input time series data
            target: Optional target for feature selection
            **kwargs: Additional arguments for extraction
            
        Returns:
            DataFrame with features (selected if target provided)
        """
        # Extract features
        features = self.extract_features(time_series_data, **kwargs)
        
        # Perform selection if target provided
        if target is not None and self.tsfresh_available:
            features = self.feature_selection(features, target)
        
        return features
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the tsfresh model."""
        return {
            "model_name": "tsfresh",
            "model_type": "Automated Time Series Feature Extraction",
            "source": "Christ et al. (2018) Neurocomputing",
            "feature_count": "1200+ (comprehensive)",
            "is_initialized": self.is_initialized,
            "tsfresh_available": self.tsfresh_available,
            "config": self.config,
            "wrapper_version": "1.0.0"
        }