import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

class CrossSubjectVisualizer:
    """Visualizer for cross-subject experiment results"""
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_per_subject_comparison(self,
                                   per_subject_results: Dict[str, Dict],
                                   train_subjects: List[str],
                                   test_subject: str,
                                   filename: str = "per_subject_comparison.png"):
        """
        Bar plot comparing accuracy across all subjects
        Highlights train vs test subjects
        """
        self.logger.info("Creating per-subject comparison plot")
        
        subjects = sorted(per_subject_results.keys())
        accuracies = [per_subject_results[s]["accuracy"] for s in subjects]
        f1_scores = [per_subject_results[s]["f1_macro"] for s in subjects]
        
        # Color code: train subjects vs test subject
        colors = []
        for s in subjects:
            if s == test_subject:
                colors.append("#F44336")  # Red for test
            elif s in train_subjects:
                colors.append("#4CAF50")  # Green for train
            else:
                colors.append("#FFC107")  # Yellow for validation
        
        fig, axes = plt.subplots(2, 1, figsize=(max(12, len(subjects) * 0.8), 10))
        
        # Accuracy plot
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(subjects)), accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xticks(range(len(subjects)))
        ax1.set_xticklabels(subjects, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Subject')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=np.mean(accuracies), color='black', linestyle='--', alpha=0.5, label='Mean')
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1-macro plot
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(subjects)), f1_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xticks(range(len(subjects)))
        ax2.set_xticklabels(subjects, rotation=45, ha='right')
        ax2.set_ylabel('F1-Macro')
        ax2.set_title('F1-Macro by Subject')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=np.mean(f1_scores), color='black', linestyle='--', alpha=0.5, label='Mean')
        
        # Add value labels on bars
        for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Legend
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='#4CAF50', edgecolor='black', label='Train'),
            Rectangle((0, 0), 1, 1, facecolor='#F44336', edgecolor='black', label='Test'),
            Rectangle((0, 0), 1, 1, facecolor='#FFC107', edgecolor='black', label='Validation')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Per-subject comparison saved: {save_path}")
    
    def plot_train_vs_test_comparison(self,
                                     training_results: Dict,
                                     test_results: Dict,
                                     filename: str = "train_vs_test_comparison.png"):
        """
        Compare performance on training subjects vs test subject
        """
        self.logger.info("Creating train vs test comparison plot")
        
        # Handle None or missing results
        if training_results is None:
            self.logger.warning("training_results is None, using empty dict")
            training_results = {}
        if test_results is None:
            self.logger.warning("test_results is None, using empty dict")
            test_results = {}
        
        metrics = ['accuracy', 'f1_macro']
        train_val_test_labels = ['Train\n(same subjects)', 'Val\n(same subjects)', 'Test\n(new subject)']
        
        # Extract metrics safely
        val_results = training_results.get('val', {})
        if val_results is None:
            val_results = {}
        train_metrics = [val_results.get(m, 0.0) for m in metrics]
        test_metrics = [test_results.get(m, 0.0) for m in metrics]
        
        # If we have training set performance
        train_results = training_results.get('train', {})
        if train_results is not None and isinstance(train_results, dict):
            train_set_metrics = [train_results.get(m, 0.0) for m in metrics]
        else:
            train_set_metrics = None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        x = np.arange(len(metrics))
        width = 0.25
        
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        if train_set_metrics:
            ax.bar(x - width, train_set_metrics, width, label=train_val_test_labels[0], 
                  color=colors[0], alpha=0.8, edgecolor='black')
            ax.bar(x, train_metrics, width, label=train_val_test_labels[1], 
                  color=colors[1], alpha=0.8, edgecolor='black')
            ax.bar(x + width, test_metrics, width, label=train_val_test_labels[2], 
                  color=colors[2], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for i, v in enumerate(train_set_metrics):
                ax.text(i - width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
            for i, v in enumerate(train_metrics):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
            for i, v in enumerate(test_metrics):
                ax.text(i + width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            ax.bar(x - width/2, train_metrics, width, label=train_val_test_labels[1], 
                  color=colors[1], alpha=0.8, edgecolor='black')
            ax.bar(x + width/2, test_metrics, width, label=train_val_test_labels[2], 
                  color=colors[2], alpha=0.8, edgecolor='black')
            
            for i, v in enumerate(train_metrics):
                ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
            for i, v in enumerate(test_metrics):
                ax.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score')
        ax.set_title('Performance: Training Subjects vs New Test Subject')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'F1-Macro'])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Train vs test comparison saved: {save_path}")
    
    def plot_cross_subject_summary(self,
                                  results: Dict,
                                  filename: str = "cross_subject_summary.png"):
        """
        Comprehensive summary of cross-subject experiment
        """
        self.logger.info("Creating cross-subject summary plot")
        
        config = results.get('config', {})
        test_results = results.get('cross_subject_test', {})
        per_subject = results.get('per_subject_analysis', {})
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Configuration summary (text)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        train_subjects_str = ', '.join(config.get('train_subjects', []))
        test_subject_str = config.get('test_subject', 'N/A')
        exercise_str = config.get('exercise', 'N/A')
        num_gestures = len(config.get('common_gestures', []))
        test_acc = test_results.get('accuracy', 0.0)
        test_f1 = test_results.get('f1_macro', 0.0)
        
        summary_text = f"""
Cross-Subject Experiment Summary
{'=' * 60}
Training Subjects: {train_subjects_str}
Test Subject: {test_subject_str}
Exercise: {exercise_str}
Common Gestures: {num_gestures} gestures
{'=' * 60}
Test Results:
  Accuracy: {test_acc:.4f}
  F1-Macro: {test_f1:.4f}
        """
        
        ax1.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 2. Per-subject accuracy
        if per_subject:
            ax2 = fig.add_subplot(gs[1, :])
            subjects = sorted(per_subject.keys())
            accuracies = [per_subject[s]['accuracy'] for s in subjects]
            colors = ['#4CAF50' if s in config.get('train_subjects', []) else '#F44336' for s in subjects]
            
            bars = ax2.bar(range(len(subjects)), accuracies, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(subjects)))
            ax2.set_xticklabels(subjects, rotation=45, ha='right')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Per-Subject Accuracy')
            ax2.set_ylim(0, 1.0)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=test_acc, color='red', linestyle='--', 
                       linewidth=2, label=f"Test Subject ({test_subject_str})")
            ax2.legend()
        
        # 3. Confusion matrix for test subject
        ax3 = fig.add_subplot(gs[2, 0])
        cm = np.array(test_results.get('confusion_matrix', [[1]]), dtype=np.float32)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax3.imshow(cm_norm, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title(f'Confusion Matrix\n(Test Subject: {test_subject_str})')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Per-class F1 scores
        ax4 = fig.add_subplot(gs[2, 1])
        
        report = test_results.get('report', {})
        class_f1s = []
        class_labels = []
        
        for i, gid in enumerate(config.get('common_gestures', [])):
            key = str(i)
            if key in report:
                class_f1s.append(report[key].get('f1-score', 0.0))
                label = "REST" if gid == 0 else f"G{gid}"
                class_labels.append(label)
        
        if class_f1s:
            ax4.barh(range(len(class_f1s)), class_f1s, color='#2196F3', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(class_labels)))
            ax4.set_yticklabels(class_labels)
            ax4.set_xlabel('F1-Score')
            ax4.set_title('Per-Class F1-Score (Test Subject)')
            ax4.set_xlim(0, 1.0)
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Cross-subject summary saved: {save_path}")
    
    def plot_data_split_schema(self,
                               cross_subject_config,
                               split_info: Dict,
                               filename: str = "data_split_schema.png"):
        """
        Visualize how data is split between train/val/test subjects
        """
        self.logger.info("Creating data split schema visualization")
        
        train_subjects = cross_subject_config.train_subjects
        test_subject = cross_subject_config.test_subject
        val_subject = cross_subject_config.val_subject
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 2], hspace=0.4)
        
        # 1. High-level subject split
        ax1 = fig.add_subplot(gs[0])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 2)
        ax1.axis('off')
        ax1.set_title('Cross-Subject Data Split Overview', fontsize=16, fontweight='bold', pad=20)
        
        # Draw subject boxes
        all_subjects = train_subjects + [test_subject]
        if val_subject and val_subject not in all_subjects:
            all_subjects.append(val_subject)
        
        n_subjects = len(all_subjects)
        box_width = 8.0 / n_subjects
        x_start = 1.0
        
        for i, subj in enumerate(all_subjects):
            x = x_start + i * box_width
            
            if subj == test_subject:
                color = '#F44336'
                label = f'{subj}\n(TEST)'
                edge_width = 3
            elif subj == val_subject:
                color = '#FFC107'
                label = f'{subj}\n(VAL)'
                edge_width = 2
            else:
                color = '#4CAF50'
                label = f'{subj}\n(TRAIN)'
                edge_width = 2
            
            box = FancyBboxPatch((x, 0.5), box_width * 0.9, 1.0,
                                boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black',
                                linewidth=edge_width, alpha=0.7)
            ax1.add_patch(box)
            ax1.text(x + box_width * 0.45, 1.0, label, 
                    ha='center', va='center', fontsize=11, fontweight='bold')
        
        # 2. Train subjects → Train/Val split detail
        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, len(train_subjects) + 1)
        ax2.axis('off')
        ax2.set_title('Training Subjects: Internal Split (Train/Val)', 
                     fontsize=14, fontweight='bold', pad=15)
        
        for i, subj in enumerate(train_subjects):
            y_pos = len(train_subjects) - i
            
            # Subject label
            ax2.text(0.5, y_pos, subj, ha='right', va='center', 
                    fontsize=11, fontweight='bold')
            
            # All repetitions bar
            total_width = 7.0
            ax2.add_patch(Rectangle((1.5, y_pos - 0.35), total_width, 0.7,
                                   facecolor='lightgray', edgecolor='black', linewidth=1.5))
            
            # If split by segments, show train/val split
            if cross_subject_config.pool_train_subjects:
                # Pooled: all subjects combined, then split
                train_ratio = 0.7  # from split_cfg
                val_ratio = 0.15
                
                train_width = total_width * train_ratio
                val_width = total_width * val_ratio
                test_width = total_width * (1 - train_ratio - val_ratio)
                
                # Train portion
                ax2.add_patch(Rectangle((1.5, y_pos - 0.35), train_width, 0.7,
                                       facecolor='#4CAF50', edgecolor='black', 
                                       linewidth=1, alpha=0.8))
                ax2.text(1.5 + train_width/2, y_pos, 'TRAIN', 
                        ha='center', va='center', fontsize=9, fontweight='bold', color='white')
                
                # Val portion
                ax2.add_patch(Rectangle((1.5 + train_width, y_pos - 0.35), val_width, 0.7,
                                       facecolor='#FFC107', edgecolor='black', 
                                       linewidth=1, alpha=0.8))
                ax2.text(1.5 + train_width + val_width/2, y_pos, 'VAL', 
                        ha='center', va='center', fontsize=9, fontweight='bold')
                
                # Test portion (not used in cross-subject)
                ax2.add_patch(Rectangle((1.5 + train_width + val_width, y_pos - 0.35), 
                                       test_width, 0.7,
                                       facecolor='lightgray', edgecolor='black', 
                                       linewidth=1, alpha=0.5))
                ax2.text(1.5 + train_width + val_width + test_width/2, y_pos, 'unused', 
                        ha='center', va='center', fontsize=8, style='italic', color='gray')
        
        # Add legend
        train_patch = mpatches.Patch(color='#4CAF50', label='Train data', alpha=0.8)
        val_patch = mpatches.Patch(color='#FFC107', label='Val data', alpha=0.8)
        unused_patch = mpatches.Patch(color='lightgray', label='Unused (would be test in single-subject)', alpha=0.5)
        ax2.legend(handles=[train_patch, val_patch, unused_patch], 
                  loc='upper right', fontsize=10)
        
        # 3. Test subject (no split)
        ax3 = fig.add_subplot(gs[2])
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 2)
        ax3.axis('off')
        ax3.set_title('Test Subject: All Data Used for Testing', 
                     fontsize=14, fontweight='bold', pad=15)
        
        y_pos = 1.0
        ax3.text(0.5, y_pos, test_subject, ha='right', va='center', 
                fontsize=11, fontweight='bold')
        
        # Full bar for test
        total_width = 7.0
        ax3.add_patch(Rectangle((1.5, y_pos - 0.35), total_width, 0.7,
                               facecolor='#F44336', edgecolor='black', 
                               linewidth=2, alpha=0.8))
        ax3.text(1.5 + total_width/2, y_pos, 'ALL DATA → TEST', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Add summary text
        summary_text = f"""
Key Points:
• Train subjects: {', '.join(train_subjects)} → Model learns from these people
• Test subject: {test_subject} → Model has NEVER seen this person
• Goal: Evaluate how well model generalizes to completely new users
        """
        ax3.text(0.5, 0.2, summary_text, ha='left', va='top', 
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Data split schema saved: {save_path}")
    
    def plot_gesture_comparison_across_subjects(self,
                                                subjects_data: Dict[str, Tuple],
                                                gesture_ids: List[int],
                                                num_windows_per_gesture: int = 2,
                                                channel_idx: int = 0,
                                                filename: str = "gesture_comparison_subjects.png"):
        """
        Compare gesture windows across different subjects
        
        Args:
            subjects_data: {subject_id: (emg, segments, grouped_windows)}
            gesture_ids: which gestures to compare
            num_windows_per_gesture: how many windows to show per gesture
            channel_idx: which EMG channel to display
        """
        self.logger.info(f"Creating gesture comparison across subjects for gestures {gesture_ids}")
        
        subjects = sorted(subjects_data.keys())
        n_subjects = len(subjects)
        n_gestures = len(gesture_ids)
        
        # Create grid: rows = subjects, cols = gestures
        fig, axes = plt.subplots(n_subjects, n_gestures, 
                                figsize=(5 * n_gestures, 3 * n_subjects),
                                squeeze=False)
        
        for subj_idx, subject_id in enumerate(subjects):
            _, _, grouped_windows = subjects_data[subject_id]
            
            for gest_idx, gesture_id in enumerate(gesture_ids):
                ax = axes[subj_idx, gest_idx]
                
                # Get windows for this gesture
                if gesture_id in grouped_windows:
                    repetitions = grouped_windows[gesture_id]
                    
                    # Collect windows from first repetition(s)
                    windows_to_plot = []
                    for rep in repetitions:
                        if len(rep) > 0:
                            # Take first few windows from this repetition
                            n_take = min(num_windows_per_gesture - len(windows_to_plot), len(rep))
                            windows_to_plot.extend(rep[:n_take])
                            if len(windows_to_plot) >= num_windows_per_gesture:
                                break
                    
                    # Plot windows
                    colors = plt.cm.tab10(np.linspace(0, 1, num_windows_per_gesture))
                    for w_idx, window in enumerate(windows_to_plot[:num_windows_per_gesture]):
                        signal = window[:, channel_idx]  # (T,)
                        time_ms = np.arange(len(signal)) / 2000 * 1000  # convert to ms
                        ax.plot(time_ms, signal, alpha=0.7, linewidth=1.5, 
                               color=colors[w_idx], label=f'Window {w_idx+1}')
                    
                    ax.set_xlabel('Time (ms)', fontsize=9)
                    ax.set_ylabel('Amplitude', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    if w_idx == 0:  # Only add legend to first gesture
                        ax.legend(fontsize=8, loc='upper right')
                else:
                    # No data for this gesture
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Titles
                if subj_idx == 0:
                    gesture_label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
                    ax.set_title(gesture_label, fontsize=12, fontweight='bold')
                
                if gest_idx == 0:
                    ax.set_ylabel(f'{subject_id}\n\nAmplitude', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Gesture Comparison Across Subjects (Channel {channel_idx})', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Gesture comparison saved: {save_path}")
    
    def plot_detailed_split_breakdown(self,
                                     split_info: Dict,
                                     cross_subject_config,
                                     filename: str = "detailed_split_breakdown.png"):
        """
        Detailed breakdown of windows per gesture per split
        """
        self.logger.info("Creating detailed split breakdown")
        
        train_subjects = cross_subject_config.train_subjects
        test_subject = cross_subject_config.test_subject
        
        splits = ['train', 'val', 'test']
        gestures = sorted(split_info['train'].get('gestures', []))
        
        if not gestures:
            self.logger.warning("No gestures found in split_info")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(gestures) * 0.5)), 
                                sharey=True)
        
        colors = {'train': '#4CAF50', 'val': '#FFC107', 'test': '#F44336'}
        
        for split_idx, split_name in enumerate(splits):
            ax = axes[split_idx]
            
            split_data = split_info.get(split_name, {})
            per_gesture = split_data.get('per_gesture', {})
            
            counts = [per_gesture.get(gid, 0) for gid in gestures]
            gesture_labels = ["REST" if g == 0 else f"G{g}" for g in gestures]
            
            y_pos = np.arange(len(gestures))
            bars = ax.barh(y_pos, counts, color=colors[split_name], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                if count > 0:
                    ax.text(count + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                           f'{count}', va='center', fontsize=9, fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(gesture_labels)
            ax.set_xlabel('Number of Windows', fontsize=11)
            ax.set_title(f'{split_name.upper()}\n({split_data.get("total_windows", 0)} total)',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add subject info
            if split_name == 'test':
                subject_text = f'Subject: {test_subject}'
            else:
                subject_text = f'Subjects: {", ".join(train_subjects)}'
            
            ax.text(0.95, 0.02, subject_text, transform=ax.transAxes,
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[0].set_ylabel('Gesture', fontsize=11, fontweight='bold')
        
        plt.suptitle('Window Distribution Across Splits and Gestures', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Detailed split breakdown saved: {save_path}")