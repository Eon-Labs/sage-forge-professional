"""
catch22 Feature Extractor Wrapper

Integrates canonical time series features from computational biology.
22 discriminative time series features selected from 7000+ in the hctsa library.

Reference: Lubba et al. (2019) "catch22: CAnonical Time-series CHaracteristics"
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

class Catch22Wrapper:
    """
    Wrapper for catch22 canonical time series feature extraction.
    
    Extracts 22 research-validated features designed to capture 
    fundamental time series characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.is_initialized = False
        self.catch22_available = False
        
        console.print("[cyan]📊 catch22 wrapper initialized[/cyan]")
    
    def initialize_model(self) -> bool:
        """Initialize catch22 with dependency checks."""
        try:
            import pycatch22
            self.catch22 = pycatch22
            self.catch22_available = True
            console.print("[green]✅ pycatch22 imported successfully[/green]")
            
        except ImportError:
            console.print("[yellow]⚠️ pycatch22 not available, using synthetic catch22-like features[/yellow]")
            self.catch22_available = False
        
        self.is_initialized = True
        return True
    
    def extract_features(self, time_series: pd.Series, feature_subset: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract catch22 features from a time series.
        
        Args:
            time_series: Input time series data
            feature_subset: Optional subset of features to extract
            
        Returns:
            Dictionary of feature names and values
        """
        if not self.is_initialized:
            self.initialize_model()
        
        # Handle NaN values
        clean_series = time_series.dropna()
        if len(clean_series) < 10:
            console.print("[yellow]⚠️ Time series too short for catch22 features[/yellow]")
            return self._get_default_features()
        
        try:
            if self.catch22_available:
                return self._extract_real_catch22(clean_series, feature_subset)
            else:
                return self._extract_synthetic_catch22(clean_series, feature_subset)
                
        except Exception as e:
            console.print(f"[yellow]⚠️ catch22 extraction failed: {e}, using fallback[/yellow]")
            return self._extract_synthetic_catch22(clean_series, feature_subset)
    
    def _extract_real_catch22(self, series: pd.Series, feature_subset: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract features using real pycatch22."""
        console.print(f"[green]🎯 Extracting real catch22 features from {len(series)} points[/green]")
        
        # Convert to numpy array
        data = series.values.astype(float)
        
        # Extract all catch22 features
        try:
            feature_values = self.catch22.catch22_all(data)
            feature_names = self.catch22.catch22_all(data, catch24=False)['names']
            
            # Create feature dictionary
            features = dict(zip(feature_names, feature_values['values']))
            
            # Filter subset if requested
            if feature_subset:
                features = {k: v for k, v in features.items() if k in feature_subset}
            
            console.print(f"[green]✅ Extracted {len(features)} real catch22 features[/green]")
            return features
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Real catch22 failed: {e}[/yellow]")
            return self._extract_synthetic_catch22(series, feature_subset)
    
    def _extract_synthetic_catch22(self, series: pd.Series, feature_subset: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract synthetic features mimicking catch22 patterns."""
        console.print(f"[yellow]🔧 Extracting synthetic catch22-like features from {len(series)} points[/yellow]")
        
        data = series.values
        features = {}
        
        # Implement approximate versions of catch22 features
        feature_calculators = {
            'DN_HistogramMode_5': lambda x: self._histogram_mode(x, 5),
            'DN_HistogramMode_10': lambda x: self._histogram_mode(x, 10),
            'CO_f1ecac': lambda x: self._first_zero_autocorr(x),
            'CO_FirstMin_ac': lambda x: self._first_min_autocorr(x),
            'CO_HistogramAMI_even_2_5': lambda x: self._histogram_ami(x),
            'CO_trev_1_num': lambda x: self._time_rev_asymmetry(x),
            'MD_hrv_classic_pnn40': lambda x: self._pnn40(x),
            'SB_BinaryStats_mean_longstretch1': lambda x: self._binary_stats(x),
            'SB_TransitionMatrix_3ac_sumdiagcov': lambda x: self._transition_matrix(x),
            'PD_PeriodicityWang_th0_01': lambda x: self._periodicity_wang(x),
            'CO_Embed2_Dist_tau_d_expfit_meandiff': lambda x: self._embed2_dist(x),
            'IN_AutoMutualInfoStats_40_gaussian_fmmi': lambda x: self._auto_mutual_info(x),
            'FC_LocalSimple_mean1_tauresrat': lambda x: self._local_simple(x),
            'DN_OutlierInclude_p_001_mdrmd': lambda x: self._outlier_include(x),
            'DN_OutlierInclude_n_001_mdrmd': lambda x: self._outlier_include_neg(x),
            'SP_Summaries_welch_rect_area_5_1': lambda x: self._spectral_area(x),
            'SB_BinaryStats_diff_longstretch0': lambda x: self._binary_diff_stats(x),
            'SB_MotifThree_quantile_hh': lambda x: self._motif_three(x),
            'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1': lambda x: self._fluct_anal(x),
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1': lambda x: self._dfa_analysis(x),
            'SP_Summaries_welch_rect_centroid': lambda x: self._spectral_centroid(x),
            'FC_LocalSimple_mean3_stderr': lambda x: self._local_simple_stderr(x)
        }
        
        # Calculate requested features or all if no subset
        target_features = feature_subset if feature_subset else list(feature_calculators.keys())
        
        for feature_name in target_features:
            if feature_name in feature_calculators:
                try:
                    features[feature_name] = feature_calculators[feature_name](data)
                except Exception as e:
                    console.print(f"[yellow]⚠️ Feature {feature_name} failed: {e}[/yellow]")
                    features[feature_name] = 0.0
            else:
                features[feature_name] = 0.0
        
        console.print(f"[green]✅ Extracted {len(features)} synthetic catch22-like features[/green]")
        return features
    
    # Synthetic feature implementations (approximations of catch22 features)
    def _histogram_mode(self, data: np.ndarray, bins: int) -> float:
        """Approximate histogram mode calculation."""
        try:
            hist, bin_edges = np.histogram(data, bins=bins)
            return float(np.argmax(hist))
        except:
            return 0.0
    
    def _first_zero_autocorr(self, data: np.ndarray) -> float:
        """First zero of autocorrelation function."""
        try:
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            zero_crossings = np.where(np.diff(np.signbit(autocorr)))[0]
            return float(zero_crossings[0] if len(zero_crossings) > 0 else len(data))
        except:
            return 0.0
    
    def _first_min_autocorr(self, data: np.ndarray) -> float:
        """First minimum of autocorrelation function."""
        try:
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            return float(np.argmin(autocorr[:min(len(autocorr), 50)]))
        except:
            return 0.0
    
    def _histogram_ami(self, data: np.ndarray) -> float:
        """Histogram-based auto-mutual information."""
        try:
            hist, _ = np.histogram(data, bins=10)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log2(hist)))
        except:
            return 0.0
    
    def _time_rev_asymmetry(self, data: np.ndarray) -> float:
        """Time reversal asymmetry statistic."""
        try:
            forward = np.mean(np.diff(data) ** 3)
            backward = np.mean(np.diff(data[::-1]) ** 3)
            return float(forward - backward)
        except:
            return 0.0
    
    def _pnn40(self, data: np.ndarray) -> float:
        """Percentage of differences > 40ms (adapted for generic data)."""
        try:
            diffs = np.abs(np.diff(data))
            threshold = np.std(data) * 0.4  # Adaptive threshold
            return float(np.sum(diffs > threshold) / len(diffs))
        except:
            return 0.0
    
    def _binary_stats(self, data: np.ndarray) -> float:
        """Binary statistics on mean-thresholded data."""
        try:
            binary = (data > np.mean(data)).astype(int)
            runs = []
            current_run = 1
            for i in range(1, len(binary)):
                if binary[i] == binary[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            return float(np.mean(runs))
        except:
            return 0.0
    
    def _transition_matrix(self, data: np.ndarray) -> float:
        """Transition matrix statistics."""
        try:
            quantiles = np.quantile(data, [0.33, 0.67])
            discrete = np.digitize(data, quantiles)
            transitions = np.zeros((3, 3))
            for i in range(len(discrete) - 1):
                transitions[discrete[i], discrete[i+1]] += 1
            return float(np.trace(transitions) / np.sum(transitions))
        except:
            return 0.0
    
    def _periodicity_wang(self, data: np.ndarray) -> float:
        """Wang periodicity measure."""
        try:
            fft_vals = np.abs(np.fft.fft(data))
            return float(np.max(fft_vals[1:len(fft_vals)//2]) / np.mean(fft_vals[1:len(fft_vals)//2]))
        except:
            return 0.0
    
    def _embed2_dist(self, data: np.ndarray) -> float:
        """2D embedding distance measure."""
        try:
            if len(data) < 3:
                return 0.0
            embedded = np.column_stack((data[:-1], data[1:]))
            distances = np.sqrt(np.sum(np.diff(embedded, axis=0)**2, axis=1))
            return float(np.mean(distances))
        except:
            return 0.0
    
    def _auto_mutual_info(self, data: np.ndarray) -> float:
        """Auto-mutual information approximation."""
        try:
            if len(data) < 2:
                return 0.0
            return float(np.corrcoef(data[:-1], data[1:])[0, 1] ** 2)
        except:
            return 0.0
    
    def _local_simple(self, data: np.ndarray) -> float:
        """Local simple statistic."""
        try:
            local_means = np.convolve(data, np.ones(3)/3, mode='valid')
            return float(np.mean(local_means))
        except:
            return 0.0
    
    def _outlier_include(self, data: np.ndarray) -> float:
        """Outlier inclusion measure."""
        try:
            threshold = np.mean(data) + 3 * np.std(data)
            return float(np.sum(data > threshold) / len(data))
        except:
            return 0.0
    
    def _outlier_include_neg(self, data: np.ndarray) -> float:
        """Negative outlier inclusion measure."""
        try:
            threshold = np.mean(data) - 3 * np.std(data)
            return float(np.sum(data < threshold) / len(data))
        except:
            return 0.0
    
    def _spectral_area(self, data: np.ndarray) -> float:
        """Spectral area measure."""
        try:
            fft_vals = np.abs(np.fft.fft(data))
            return float(np.sum(fft_vals[:len(fft_vals)//2]))
        except:
            return 0.0
    
    def _binary_diff_stats(self, data: np.ndarray) -> float:
        """Binary difference statistics."""
        try:
            diffs = np.diff(data)
            binary_diffs = (diffs > 0).astype(int)
            return float(np.mean(binary_diffs))
        except:
            return 0.0
    
    def _motif_three(self, data: np.ndarray) -> float:
        """Three-point motif analysis."""
        try:
            if len(data) < 3:
                return 0.0
            motifs = []
            for i in range(len(data) - 2):
                triplet = data[i:i+3]
                motifs.append(np.sum(triplet))
            return float(np.std(motifs))
        except:
            return 0.0
    
    def _fluct_anal(self, data: np.ndarray) -> float:
        """Fluctuation analysis approximation."""
        try:
            detrended = data - np.mean(data)
            return float(np.std(detrended) / np.mean(np.abs(detrended)))
        except:
            return 0.0
    
    def _dfa_analysis(self, data: np.ndarray) -> float:
        """Detrended fluctuation analysis approximation."""
        try:
            integrated = np.cumsum(data - np.mean(data))
            return float(np.std(integrated) / len(integrated))
        except:
            return 0.0
    
    def _spectral_centroid(self, data: np.ndarray) -> float:
        """Spectral centroid calculation."""
        try:
            fft_vals = np.abs(np.fft.fft(data))
            freqs = np.fft.fftfreq(len(data))
            return float(np.sum(freqs[:len(freqs)//2] * fft_vals[:len(fft_vals)//2]) / 
                        np.sum(fft_vals[:len(fft_vals)//2]))
        except:
            return 0.0
    
    def _local_simple_stderr(self, data: np.ndarray) -> float:
        """Local simple standard error."""
        try:
            window_size = min(5, len(data)//4)
            if window_size < 2:
                return 0.0
            local_stds = []
            for i in range(len(data) - window_size + 1):
                window = data[i:i+window_size]
                local_stds.append(np.std(window))
            return float(np.std(local_stds))
        except:
            return 0.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when extraction fails."""
        return {f"catch22_feature_{i+1}": 0.0 for i in range(22)}
    
    def extract_features_dataframe(self, data: pd.DataFrame, 
                                 column: str = 'close',
                                 window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Extract catch22 features from DataFrame with optional rolling windows.
        
        Args:
            data: Input DataFrame with time series data
            column: Column name to extract features from
            window_size: Optional rolling window size
            
        Returns:
            DataFrame with catch22 features as columns
        """
        if column not in data.columns:
            console.print(f"[red]❌ Column {column} not found in data[/red]")
            return pd.DataFrame()
        
        series = data[column]
        
        if window_size is None:
            # Extract features from entire series
            features = self.extract_features(series)
            feature_df = pd.DataFrame([features], index=[data.index[-1]])
        else:
            # Extract features using rolling windows
            console.print(f"[cyan]🔄 Extracting rolling catch22 features (window={window_size})[/cyan]")
            
            results = []
            valid_indices = []
            
            for i in range(window_size, len(series)):
                window_data = series.iloc[i-window_size:i]
                features = self.extract_features(window_data)
                results.append(features)
                valid_indices.append(data.index[i])
            
            feature_df = pd.DataFrame(results, index=valid_indices)
        
        console.print(f"[green]✅ Generated catch22 feature DataFrame: {feature_df.shape}[/green]")
        return feature_df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all catch22 features."""
        return {
            'DN_HistogramMode_5': 'Mode of histogram with 5 bins',
            'DN_HistogramMode_10': 'Mode of histogram with 10 bins', 
            'CO_f1ecac': 'First zero crossing of autocorrelation function',
            'CO_FirstMin_ac': 'First minimum of autocorrelation function',
            'CO_HistogramAMI_even_2_5': 'Histogram-based auto-mutual information',
            'CO_trev_1_num': 'Time reversal asymmetry statistic',
            'MD_hrv_classic_pnn40': 'Percentage of differences > threshold',
            'SB_BinaryStats_mean_longstretch1': 'Mean length of binary stretches',
            'SB_TransitionMatrix_3ac_sumdiagcov': 'Transition matrix diagonal coverage',
            'PD_PeriodicityWang_th0_01': 'Wang periodicity measure',
            'CO_Embed2_Dist_tau_d_expfit_meandiff': '2D embedding distance statistic',
            'IN_AutoMutualInfoStats_40_gaussian_fmmi': 'Auto-mutual information statistic',
            'FC_LocalSimple_mean1_tauresrat': 'Local simple forecast accuracy',
            'DN_OutlierInclude_p_001_mdrmd': 'Positive outlier inclusion measure',
            'DN_OutlierInclude_n_001_mdrmd': 'Negative outlier inclusion measure',
            'SP_Summaries_welch_rect_area_5_1': 'Spectral power area measure',
            'SB_BinaryStats_diff_longstretch0': 'Binary difference statistics',
            'SB_MotifThree_quantile_hh': 'Three-point motif quantile measure',
            'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1': 'Fluctuation analysis statistic',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1': 'Detrended fluctuation analysis',
            'SP_Summaries_welch_rect_centroid': 'Spectral centroid measure',
            'FC_LocalSimple_mean3_stderr': 'Local forecasting standard error'
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the catch22 model."""
        return {
            "model_name": "catch22",
            "model_type": "Canonical Time Series Features",
            "source": "Lubba et al. (2019) Nature Communications",
            "feature_count": 22,
            "is_initialized": self.is_initialized,
            "pycatch22_available": self.catch22_available,
            "config": self.config,
            "wrapper_version": "1.0.0"
        }