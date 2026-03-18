"""
Validation of data quality improvements.

Compares data quality BEFORE and AFTER applying improved processing
to verify that improvements are effective.
"""

import numpy as np
import logging
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, Optional


class ImprovementValidator:
    """Validates effectiveness of data improvements."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def validate_all(self, 
                    splits_before: dict,
                    splits_after: dict,
                    class_names: dict):
        """
        Compare BEFORE and AFTER improvements.
        
        Args:
            splits_before: Data BEFORE processing
            splits_after: Data AFTER processing
            class_names: {gid: name}
        """
        self.logger.info("=" * 80)
        self.logger.info("VALIDATION OF IMPROVEMENTS")
        self.logger.info("=" * 80)
        
        # 1. Compare separability
        self._compare_separability(splits_before, splits_after)
        
        # 2. Compare signal quality
        self._compare_signal_quality(splits_before, splits_after)
        
        # 3. Baseline model (Logistic Regression)
        self._compare_baseline_performance(splits_before, splits_after)
        
        self.logger.info("=" * 80)
    
    def _compare_separability(self, splits_before: dict, splits_after: dict):
        """Compare Silhouette Score."""
        self.logger.info("\n[1] Separability Comparison")
        
        def compute_silhouette(splits: dict) -> float:
            X_list, y_list = [], []
            class_ids = sorted(splits['train'].keys())
            
            for i, gid in enumerate(class_ids):
                if gid in splits['train']:
                    windows = splits['train'][gid]
                    if len(windows) > 0:
                        # If features (2D), use as is
                        # If windows (3D), take mean
                        if windows.ndim == 3:
                            features = windows.mean(axis=1)
                        else:
                            features = windows
                        
                        X_list.append(features)
                        y_list.append(np.full(len(features), i))
            
            if len(X_list) < 2:
                return np.nan
            
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            
            # Limit size for speed
            if len(X) > 5000:
                indices = np.random.choice(len(X), 5000, replace=False)
                X = X[indices]
                y = y[indices]
            
            try:
                score = silhouette_score(X, y, metric='euclidean')
                return score
            except (ValueError, RuntimeError):
                return np.nan
        
        score_before = compute_silhouette(splits_before)
        score_after = compute_silhouette(splits_after)
        
        self.logger.info(f"  Silhouette Score BEFORE: {score_before:.3f}")
        self.logger.info(f"  Silhouette Score AFTER:  {score_after:.3f}")
        
        if not np.isnan(score_before) and not np.isnan(score_after):
            improvement = score_after - score_before
            self.logger.info(f"  Improvement: {improvement:+.3f}")
            
            if improvement > 0.1:
                self.logger.info("  ✓ Significant improvement!")
            elif improvement > 0:
                self.logger.info("  ✓ Small improvement")
            else:
                self.logger.warning("  ⚠️ Degradation or no change")
    
    def _compare_signal_quality(self, splits_before: dict, splits_after: dict):
        """Compare signal quality (saturation)."""
        self.logger.info("\n[2] Signal Quality Comparison")
        
        def compute_saturation(splits: dict) -> float:
            """Percentage of saturated samples."""
            all_data = []
            for windows in splits['train'].values():
                if len(windows) > 0 and windows.ndim == 3:
                    all_data.append(windows[:100])  # First 100 windows
            
            if not all_data:
                return np.nan
            
            data = np.concatenate(all_data, axis=0)
            
            # Count saturation as > 2.8 std
            std = np.std(data)
            saturated = np.sum(np.abs(data) > 2.8 * std)
            return 100 * saturated / data.size
        
        sat_before = compute_saturation(splits_before)
        sat_after = compute_saturation(splits_after)
        
        self.logger.info(f"  Saturation BEFORE: {sat_before:.2f}%")
        self.logger.info(f"  Saturation AFTER:  {sat_after:.2f}%")
        
        if not np.isnan(sat_before) and not np.isnan(sat_after):
            reduction = sat_before - sat_after
            self.logger.info(f"  Reduction: {reduction:.2f}%")
            
            if reduction > 1.0:
                self.logger.info("  ✓ Saturation significantly reduced!")
            elif reduction > 0:
                self.logger.info("  ✓ Saturation reduced")
            else:
                self.logger.warning("  ⚠️ Saturation not changed")
    
    def _compare_baseline_performance(self, splits_before: dict, splits_after: dict):
        """Compare simple model performance."""
        self.logger.info("\n[3] Baseline Model Comparison (Logistic Regression)")
        
        def train_and_test(splits: dict) -> float:
            """Train LR and return test accuracy."""
            X_train_list, y_train_list = [], []
            X_test_list, y_test_list = [], []
            
            class_ids = sorted(splits['train'].keys())
            
            for i, gid in enumerate(class_ids):
                # Train
                if gid in splits['train']:
                    windows = splits['train'][gid]
                    if len(windows) > 0:
                        if windows.ndim == 3:
                            features = windows.mean(axis=1)
                        else:
                            features = windows
                        
                        X_train_list.append(features)
                        y_train_list.append(np.full(len(features), i))
                
                # Test
                if gid in splits['test']:
                    windows = splits['test'][gid]
                    if len(windows) > 0:
                        if windows.ndim == 3:
                            features = windows.mean(axis=1)
                        else:
                            features = windows
                        
                        X_test_list.append(features)
                        y_test_list.append(np.full(len(features), i))
            
            if len(X_train_list) < 2 or len(X_test_list) < 2:
                return np.nan
            
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
            
            # Train
            lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
            lr.fit(X_train, y_train)
            
            # Test
            y_pred = lr.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            return acc
        
        acc_before = train_and_test(splits_before)
        acc_after = train_and_test(splits_after)
        
        self.logger.info(f"  Test Accuracy BEFORE: {acc_before*100:.2f}%")
        self.logger.info(f"  Test Accuracy AFTER:  {acc_after*100:.2f}%")
        
        if not np.isnan(acc_before) and not np.isnan(acc_after):
            improvement = (acc_after - acc_before) * 100
            self.logger.info(f"  Improvement: {improvement:+.2f}%")
            
            if improvement > 10:
                self.logger.info("  ✓ Significant improvement!")
            elif improvement > 5:
                self.logger.info("  ✓ Notable improvement")
            elif improvement > 0:
                self.logger.info("  ✓ Small improvement")
            else:
                self.logger.warning("  ⚠️ No improvement or degradation")