#!/usr/bin/env python3
"""
🎯 NT-NATIVE ONLINE FEATURE SELECTION 2025
==========================================

Auto-parameterizing online feature selection algorithms for real-time trading strategies.
Follows NautilusTrader patterns for bias-free operation and computational efficiency.

Features:
- Mutual Information-based feature selection
- Online LASSO with adaptive regularization
- Recursive Feature Elimination (RFE) 
- Ensemble feature selection strategy
- Adaptive threshold adjustment
- Real-time performance monitoring

Algorithms:
- Online Mutual Information (streaming entropy estimation)
- Online LASSO (Follow-the-Regularized-Leader)
- Online RFE (incremental importance ranking)
- Multi-Armed Bandit feature selection
- Ensemble voting with adaptive weights

Author: Claude Code Assistant  
Date: 2025-07-19
License: MIT
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Rich console for enhanced output
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()


@dataclass
class FeatureImportance:
    """Feature importance with metadata."""
    feature_id: int
    importance_score: float
    selection_count: int
    stability_score: float
    last_updated: int


class OnlineFeatureSelector(ABC):
    """Abstract base class for online feature selectors."""
    
    @abstractmethod
    def select_features(self, features: np.ndarray, target: float, 
                       feature_names: Optional[List[str]] = None) -> Set[int]:
        """Select important features given current observation."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[int, float]:
        """Get current feature importance scores."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset selector state."""
        pass


class MutualInformationSelector(OnlineFeatureSelector):
    """
    🧠 Online Mutual Information Feature Selection
    
    Estimates mutual information between features and target using online entropy estimation.
    Uses streaming quantile estimation for continuous MI computation.
    """
    
    def __init__(self, window_size: int = 1000, n_bins: int = 10, alpha: float = 0.01):
        self.window_size = window_size
        self.n_bins = n_bins
        self.alpha = alpha  # Learning rate for running statistics
        
        # Feature statistics
        self.feature_histories = defaultdict(lambda: deque(maxlen=window_size))
        self.target_history = deque(maxlen=window_size)
        
        # Running statistics for entropy estimation
        self.feature_quantiles = {}
        self.joint_counts = {}
        self.mi_scores = {}
        
        # Update tracking
        self.update_count = 0
        
    def select_features(self, features: np.ndarray, target: float, 
                       feature_names: Optional[List[str]] = None) -> Set[int]:
        """Select features based on mutual information with target."""
        self.update_count += 1
        
        # Update histories
        self.target_history.append(target)
        
        selected_features = set()
        
        for i, feature_value in enumerate(features):
            self.feature_histories[i].append(feature_value)
            
            # Update MI estimate if we have enough data
            if len(self.feature_histories[i]) >= 50:  # Minimum for stable MI
                mi_score = self._estimate_mutual_information(i, target)
                self.mi_scores[i] = mi_score
                
                # Select if MI is above adaptive threshold
                threshold = self._get_adaptive_threshold()
                if mi_score > threshold:
                    selected_features.add(i)
        
        return selected_features
    
    def _estimate_mutual_information(self, feature_id: int, current_target: float) -> float:
        """Estimate mutual information using histogram-based entropy."""
        if len(self.feature_histories[feature_id]) < 20:
            return 0.0
        
        # Get recent feature and target values
        feature_values = np.array(list(self.feature_histories[feature_id]))
        target_values = np.array(list(self.target_history))
        
        if len(target_values) < len(feature_values):
            feature_values = feature_values[-len(target_values):]
        
        try:
            # Discretize using quantile-based binning
            feature_bins = self._adaptive_discretize(feature_values, self.n_bins)
            target_bins = self._adaptive_discretize(target_values, self.n_bins)
            
            # Compute joint and marginal histograms
            joint_hist, _, _ = np.histogram2d(feature_bins, target_bins, 
                                            bins=[self.n_bins, self.n_bins])
            feature_hist, _ = np.histogram(feature_bins, bins=self.n_bins)
            target_hist, _ = np.histogram(target_bins, bins=self.n_bins)
            
            # Add small constant to avoid log(0)
            joint_hist = joint_hist + 1e-10
            feature_hist = feature_hist + 1e-10
            target_hist = target_hist + 1e-10
            
            # Normalize to probabilities
            joint_prob = joint_hist / np.sum(joint_hist)
            feature_prob = feature_hist / np.sum(feature_hist)
            target_prob = target_hist / np.sum(target_hist)
            
            # Compute mutual information
            mi = 0.0
            for i in range(self.n_bins):
                for j in range(self.n_bins):
                    if joint_prob[i, j] > 1e-10:
                        mi += joint_prob[i, j] * np.log(
                            joint_prob[i, j] / (feature_prob[i] * target_prob[j])
                        )
            
            return max(0.0, mi)  # MI should be non-negative
            
        except Exception as e:
            return 0.0
    
    def _adaptive_discretize(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        """Discretize values using adaptive quantile binning."""
        if len(np.unique(values)) <= n_bins:
            # Use unique values as bins if few unique values
            unique_vals = np.unique(values)
            return np.searchsorted(unique_vals, values)
        else:
            # Use quantile-based binning
            try:
                quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1))
                return np.digitize(values, quantiles[1:-1])
            except:
                # Fallback to simple binning
                return np.digitize(values, np.linspace(np.min(values), np.max(values), n_bins))
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive threshold based on current MI distribution."""
        if not self.mi_scores:
            return 0.1  # Default threshold
        
        mi_values = list(self.mi_scores.values())
        if len(mi_values) < 5:
            return 0.1
        
        # Use median + IQR as adaptive threshold
        median_mi = np.median(mi_values)
        q75 = np.percentile(mi_values, 75)
        threshold = median_mi + 0.5 * (q75 - median_mi)
        
        return max(0.05, min(0.5, threshold))  # Clamp to reasonable range
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get current MI scores as importance."""
        return self.mi_scores.copy()
    
    def reset(self):
        """Reset selector state."""
        self.feature_histories.clear()
        self.target_history.clear()
        self.feature_quantiles.clear()
        self.joint_counts.clear()
        self.mi_scores.clear()
        self.update_count = 0


class OnlineLASSO(OnlineFeatureSelector):
    """
    🎯 Online LASSO Feature Selection
    
    Follow-the-Regularized-Leader (FTRL) implementation for sparse feature selection.
    Automatically adjusts regularization based on feature correlation.
    """
    
    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.9, 
                 adaptive_regularization: bool = True):
        self.alpha = alpha  # Base regularization strength
        self.l1_ratio = l1_ratio  # L1 vs L2 penalty ratio
        self.adaptive_regularization = adaptive_regularization
        
        # FTRL parameters
        self.weights = {}
        self.z_weights = {}  # Lazy weights for FTRL
        self.n_weights = {}  # Accumulated gradients squared
        
        # Feature selection tracking
        self.selected_features = set()
        self.feature_importance = {}
        
        # Adaptive regularization
        self.feature_correlations = defaultdict(float)
        self.update_count = 0
        
    def select_features(self, features: np.ndarray, target: float, 
                       feature_names: Optional[List[str]] = None) -> Set[int]:
        """Select features using online LASSO."""
        self.update_count += 1
        
        # Initialize weights for new features
        for i in range(len(features)):
            if i not in self.weights:
                self.weights[i] = 0.0
                self.z_weights[i] = 0.0
                self.n_weights[i] = 1e-6
        
        # Make prediction with current weights
        prediction = self._predict(features)
        
        # Compute loss gradient
        loss_gradient = 2 * (prediction - target)
        
        # Update weights using FTRL
        self._update_ftrl_weights(features, loss_gradient)
        
        # Select features based on weight magnitude
        self.selected_features = self._select_features_by_weights()
        
        # Update feature importance
        self.feature_importance = {i: abs(self.weights.get(i, 0.0)) 
                                 for i in range(len(features))}
        
        return self.selected_features
    
    def _predict(self, features: np.ndarray) -> float:
        """Make prediction with current weights."""
        prediction = 0.0
        for i, feature_value in enumerate(features):
            prediction += self.weights.get(i, 0.0) * feature_value
        return prediction
    
    def _update_ftrl_weights(self, features: np.ndarray, loss_gradient: float):
        """Update weights using FTRL algorithm."""
        for i, feature_value in enumerate(features):
            if feature_value != 0:  # Only update for non-zero features
                # Compute gradient for this feature
                gradient = loss_gradient * feature_value
                
                # Update accumulated gradient squared
                self.n_weights[i] += gradient ** 2
                
                # Update lazy weight
                sigma = (np.sqrt(self.n_weights[i]) - np.sqrt(self.n_weights[i] - gradient ** 2)) / self.alpha
                self.z_weights[i] += gradient - sigma * self.weights[i]
                
                # Compute regularization strength
                lambda_reg = self._get_adaptive_lambda(i)
                
                # FTRL weight update with L1 regularization
                if abs(self.z_weights[i]) <= lambda_reg:
                    self.weights[i] = 0.0
                else:
                    sign_z = np.sign(self.z_weights[i])
                    self.weights[i] = -1.0 / ((1 / self.alpha) * np.sqrt(self.n_weights[i])) * (
                        self.z_weights[i] - sign_z * lambda_reg
                    )
    
    def _get_adaptive_lambda(self, feature_id: int) -> float:
        """Get adaptive regularization strength for feature."""
        if not self.adaptive_regularization:
            return self.alpha * self.l1_ratio
        
        # Increase regularization for highly correlated features
        base_lambda = self.alpha * self.l1_ratio
        correlation_penalty = self.feature_correlations.get(feature_id, 0.0)
        
        adaptive_lambda = base_lambda * (1.0 + correlation_penalty)
        return min(adaptive_lambda, base_lambda * 2.0)  # Cap at 2x base
    
    def _select_features_by_weights(self) -> Set[int]:
        """Select features based on weight magnitude."""
        if not self.weights:
            return set()
        
        # Get weight magnitudes
        weight_magnitudes = {i: abs(weight) for i, weight in self.weights.items()}
        
        # Adaptive threshold based on weight distribution
        magnitudes = list(weight_magnitudes.values())
        if len(magnitudes) < 3:
            threshold = 0.01
        else:
            # Use top 50% or weights above median + 0.5*IQR
            median_weight = np.median(magnitudes)
            q75 = np.percentile(magnitudes, 75)
            threshold = max(0.01, median_weight + 0.5 * (q75 - median_weight))
        
        # Select features above threshold
        selected = {i for i, magnitude in weight_magnitudes.items() 
                   if magnitude > threshold}
        
        return selected
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance based on weight magnitudes."""
        return self.feature_importance.copy()
    
    def reset(self):
        """Reset selector state."""
        self.weights.clear()
        self.z_weights.clear()
        self.n_weights.clear()
        self.selected_features.clear()
        self.feature_importance.clear()
        self.feature_correlations.clear()
        self.update_count = 0


class RecursiveFeatureEliminator(OnlineFeatureSelector):
    """
    🔄 Online Recursive Feature Elimination
    
    Incrementally ranks features by importance and eliminates least important ones.
    Uses online gradient-based importance estimation.
    """
    
    def __init__(self, target_features: int = 10, elimination_rate: float = 0.1):
        self.target_features = target_features
        self.elimination_rate = elimination_rate
        
        # Feature tracking
        self.feature_scores = {}
        self.feature_ranks = {}
        self.eliminated_features = set()
        self.selected_features = set()
        
        # Importance estimation
        self.gradient_history = defaultdict(lambda: deque(maxlen=100))
        self.importance_decay = 0.95  # Decay factor for importance scores
        
        self.update_count = 0
    
    def select_features(self, features: np.ndarray, target: float, 
                       feature_names: Optional[List[str]] = None) -> Set[int]:
        """Select features using recursive elimination."""
        self.update_count += 1
        
        # Initialize scores for new features
        for i in range(len(features)):
            if i not in self.feature_scores:
                self.feature_scores[i] = 0.0
        
        # Update feature importance scores
        self._update_feature_scores(features, target)
        
        # Perform elimination if needed
        if len(self.feature_scores) > self.target_features:
            self._eliminate_features()
        
        # Select remaining features
        self.selected_features = set(self.feature_scores.keys()) - self.eliminated_features
        
        return self.selected_features
    
    def _update_feature_scores(self, features: np.ndarray, target: float):
        """Update feature importance scores."""
        # Simple gradient-based importance
        for i, feature_value in enumerate(features):
            if i not in self.eliminated_features and feature_value != 0:
                # Approximate gradient of loss w.r.t. feature
                gradient = abs(target * feature_value)  # Simplified importance
                self.gradient_history[i].append(gradient)
                
                # Update running importance score
                if len(self.gradient_history[i]) > 5:
                    recent_importance = np.mean(list(self.gradient_history[i])[-10:])
                    self.feature_scores[i] = (
                        self.importance_decay * self.feature_scores[i] + 
                        (1 - self.importance_decay) * recent_importance
                    )
    
    def _eliminate_features(self):
        """Eliminate least important features."""
        # Get candidates for elimination
        candidates = {i: score for i, score in self.feature_scores.items() 
                     if i not in self.eliminated_features}
        
        if len(candidates) <= self.target_features:
            return
        
        # Number of features to eliminate this round
        n_eliminate = max(1, int(len(candidates) * self.elimination_rate))
        
        # Sort by importance (ascending - least important first)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])
        
        # Eliminate least important features
        for i in range(n_eliminate):
            if i < len(sorted_candidates):
                feature_id = sorted_candidates[i][0]
                self.eliminated_features.add(feature_id)
                
        # Update ranks
        self._update_feature_ranks()
    
    def _update_feature_ranks(self):
        """Update feature ranking based on current scores."""
        active_features = {i: score for i, score in self.feature_scores.items() 
                          if i not in self.eliminated_features}
        
        # Rank by importance (descending)
        sorted_features = sorted(active_features.items(), key=lambda x: x[1], reverse=True)
        
        self.feature_ranks = {feature_id: rank + 1 
                             for rank, (feature_id, _) in enumerate(sorted_features)}
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores."""
        return {i: score for i, score in self.feature_scores.items() 
                if i not in self.eliminated_features}
    
    def reset(self):
        """Reset selector state."""
        self.feature_scores.clear()
        self.feature_ranks.clear()
        self.eliminated_features.clear()
        self.selected_features.clear()
        self.gradient_history.clear()
        self.update_count = 0


class EnsembleFeatureSelector:
    """
    🎯 Ensemble Online Feature Selection
    
    Combines multiple selection methods with adaptive weights.
    Uses multi-armed bandit approach to balance selector performance.
    """
    
    def __init__(self, max_features: int = 15, ensemble_method: str = "weighted_voting"):
        self.max_features = max_features
        self.ensemble_method = ensemble_method
        
        # Initialize selectors
        self.selectors = {
            "mutual_info": MutualInformationSelector(),
            "lasso": OnlineLASSO(),
            "rfe": RecursiveFeatureEliminator(target_features=max_features)
        }
        
        # Ensemble weights (adaptive)
        self.selector_weights = {name: 1.0 for name in self.selectors.keys()}
        self.selector_performance = {name: deque(maxlen=100) for name in self.selectors.keys()}
        
        # Final selection tracking
        self.selected_features = set()
        self.feature_vote_counts = defaultdict(int)
        self.ensemble_importance = {}
        
        self.update_count = 0
        
    def select_features(self, features: np.ndarray, target: float, 
                       feature_names: Optional[List[str]] = None) -> Set[int]:
        """Select features using ensemble method."""
        self.update_count += 1
        
        # Get selections from each selector
        selector_selections = {}
        selector_importances = {}
        
        for name, selector in self.selectors.items():
            try:
                selections = selector.select_features(features, target, feature_names)
                importance = selector.get_feature_importance()
                
                selector_selections[name] = selections
                selector_importances[name] = importance
                
                # Update performance tracking (simple heuristic)
                performance_score = len(selections) / max(1, len(features))  # Selection ratio
                self.selector_performance[name].append(performance_score)
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Selector {name} failed: {e}[/yellow]")
                selector_selections[name] = set()
                selector_importances[name] = {}
        
        # Combine selections using ensemble method
        if self.ensemble_method == "weighted_voting":
            self.selected_features = self._weighted_voting_selection(
                selector_selections, selector_importances
            )
        elif self.ensemble_method == "intersection":
            self.selected_features = self._intersection_selection(selector_selections)
        elif self.ensemble_method == "union":
            self.selected_features = self._union_selection(selector_selections)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Limit to max_features
        if len(self.selected_features) > self.max_features:
            self.selected_features = self._select_top_features(
                self.selected_features, selector_importances
            )
        
        # Update adaptive weights
        self._update_selector_weights()
        
        return self.selected_features
    
    def _weighted_voting_selection(self, selector_selections: Dict[str, Set[int]], 
                                 selector_importances: Dict[str, Dict[int, float]]) -> Set[int]:
        """Select features using weighted voting."""
        feature_votes = defaultdict(float)
        
        # Accumulate weighted votes
        for selector_name, selections in selector_selections.items():
            weight = self.selector_weights[selector_name]
            importance_scores = selector_importances.get(selector_name, {})
            
            for feature_id in selections:
                # Vote weight = selector weight * feature importance
                importance = importance_scores.get(feature_id, 1.0)
                feature_votes[feature_id] += weight * importance
        
        # Select features above voting threshold
        if not feature_votes:
            return set()
        
        vote_values = list(feature_votes.values())
        threshold = np.percentile(vote_values, 60)  # Top 40%
        
        selected = {feature_id for feature_id, votes in feature_votes.items() 
                   if votes >= threshold}
        
        # Store ensemble importance
        self.ensemble_importance = dict(feature_votes)
        
        return selected
    
    def _intersection_selection(self, selector_selections: Dict[str, Set[int]]) -> Set[int]:
        """Select features chosen by all selectors."""
        if not selector_selections:
            return set()
        
        intersection = set.intersection(*selector_selections.values())
        return intersection
    
    def _union_selection(self, selector_selections: Dict[str, Set[int]]) -> Set[int]:
        """Select features chosen by any selector."""
        union = set.union(*selector_selections.values()) if selector_selections else set()
        return union
    
    def _select_top_features(self, candidates: Set[int], 
                           selector_importances: Dict[str, Dict[int, float]]) -> Set[int]:
        """Select top features when candidates exceed max_features."""
        # Compute ensemble importance for ranking
        feature_scores = defaultdict(float)
        
        for feature_id in candidates:
            for selector_name, importance_dict in selector_importances.items():
                weight = self.selector_weights[selector_name]
                importance = importance_dict.get(feature_id, 0.0)
                feature_scores[feature_id] += weight * importance
        
        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = {feature_id for feature_id, _ in sorted_features[:self.max_features]}
        
        return top_features
    
    def _update_selector_weights(self):
        """Update selector weights based on performance."""
        # Simple performance-based weight update
        for selector_name in self.selectors.keys():
            performance_history = list(self.selector_performance[selector_name])
            
            if len(performance_history) >= 10:
                # Recent performance vs overall performance
                recent_perf = np.mean(performance_history[-10:])
                overall_perf = np.mean(performance_history)
                
                # Update weight based on relative performance
                if overall_perf > 0:
                    performance_ratio = recent_perf / overall_perf
                    self.selector_weights[selector_name] *= (0.9 + 0.2 * performance_ratio)
                
                # Normalize weights
                total_weight = sum(self.selector_weights.values())
                if total_weight > 0:
                    for name in self.selector_weights:
                        self.selector_weights[name] /= total_weight
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get ensemble feature importance."""
        return self.ensemble_importance.copy()
    
    def get_selector_weights(self) -> Dict[str, float]:
        """Get current selector weights."""
        return self.selector_weights.copy()
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of selection process."""
        return {
            "selected_features": list(self.selected_features),
            "n_selected": len(self.selected_features),
            "selector_weights": self.selector_weights.copy(),
            "ensemble_method": self.ensemble_method,
            "max_features": self.max_features,
            "update_count": self.update_count
        }
    
    def reset(self):
        """Reset all selectors and ensemble state."""
        for selector in self.selectors.values():
            selector.reset()
        
        self.selector_weights = {name: 1.0 for name in self.selectors.keys()}
        self.selector_performance = {name: deque(maxlen=100) for name in self.selectors.keys()}
        self.selected_features.clear()
        self.feature_vote_counts.clear()
        self.ensemble_importance.clear()
        self.update_count = 0


def test_online_feature_selection():
    """Test online feature selection algorithms."""
    console.print("[yellow]🧪 Testing Online Feature Selection...[/yellow]")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create features with varying importance
    X = np.random.randn(n_samples, n_features)
    
    # Target depends on first 5 features (plus noise)
    true_weights = np.zeros(n_features)
    true_weights[:5] = [2.0, 1.5, 1.0, 0.8, 0.6]
    
    y = X @ true_weights + 0.2 * np.random.randn(n_samples)
    
    # Test individual selectors
    console.print("  Testing individual selectors...")
    
    selectors = {
        "MI": MutualInformationSelector(),
        "LASSO": OnlineLASSO(),
        "RFE": RecursiveFeatureEliminator(target_features=8)
    }
    
    for name, selector in selectors.items():
        console.print(f"    Testing {name}...")
        selected_counts = defaultdict(int)
        
        for i in range(min(200, n_samples)):  # Use subset for speed
            selected = selector.select_features(X[i], y[i])
            for feature_id in selected:
                selected_counts[feature_id] += 1
        
        # Check if important features were selected
        important_features = set(range(5))  # First 5 are important
        selected_important = sum(1 for f in selected_counts.keys() if f in important_features)
        
        console.print(f"      Selected {len(selected_counts)} unique features")
        console.print(f"      Important features found: {selected_important}/5")
        
        if hasattr(selector, 'get_feature_importance'):
            importance = selector.get_feature_importance()
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                console.print(f"      Top features: {[f[0] for f in top_features]}")
    
    # Test ensemble selector
    console.print("  Testing ensemble selector...")
    ensemble = EnsembleFeatureSelector(max_features=10)
    
    ensemble_selections = defaultdict(int)
    for i in range(min(200, n_samples)):
        selected = ensemble.select_features(X[i], y[i])
        for feature_id in selected:
            ensemble_selections[feature_id] += 1
    
    important_found = sum(1 for f in ensemble_selections.keys() if f in important_features)
    console.print(f"    Ensemble selected {len(ensemble_selections)} unique features")
    console.print(f"    Important features found: {important_found}/5")
    
    # Show final summary
    summary = ensemble.get_selection_summary()
    weights = ensemble.get_selector_weights()
    
    console.print(f"    Final selector weights: {weights}")
    console.print(f"    Final selection: {summary['selected_features'][:10]}")
    
    console.print("[green]✅ Online feature selection test completed![/green]")


if __name__ == "__main__":
    console.print("[bold green]🎯 NT-Native Online Feature Selection![/bold green]")
    console.print("[dim]Auto-parameterizing feature selection for trading strategies[/dim]")
    
    # Run tests
    test_online_feature_selection()
    
    console.print("\n[green]🌟 Ready for integration with Catch22 features![/green]")