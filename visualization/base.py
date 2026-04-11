import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
from sklearn.preprocessing import label_binarize
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, roc_curve, auc
from config.base import ProcessingConfig

class Visualizer:
    """Visualizer results"""
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_segments_split_timeline(
        self,
        emg: np.ndarray,
        stimulus: np.ndarray,
        assignments: Dict[int, List[List[str]]],
        filename: str = "segments_split_timeline.png",
        sampling_rate: int = 2000,
    ):
        """
        Displays a timeline of segments, colored by their membership in train/val/test.
        If a segment contains mixed labels (in by_windows mode), the majority is used.
        REST segments (gid=0) are grayed out.
        """
        self.logger.info("Visualization: timeline of segmentation (train/val/test)")

        subset_colors = {"train": "#4CAF50", "val": "#FFC107", "test": "#F44336", "none": "#BDBDBD"}

        stim = stimulus.flatten()
        sr = sampling_rate
        change_idxs = np.where(np.diff(stim, prepend=stim[0]) != 0)[0]
        segments_timeline: List[Tuple[int, int, int]] = []
        for i in range(len(change_idxs)):
            start = change_idxs[i]
            end = change_idxs[i + 1] if i + 1 < len(change_idxs) else len(stim)
            gid = int(stim[start])
            segments_timeline.append((start, end, gid))

        occ_counter: Dict[int, int] = {}
        seg_labels: List[Tuple[int, int, int, str]] = []  # (start, end, gid, lbl)

        for (s_idx, e_idx, gid) in segments_timeline:
            occ = occ_counter.get(gid, 0)
            occ_counter[gid] = occ + 1

            lbl = "none"
            if gid in assignments and occ < len(assignments[gid]):
                labels_for_seg = assignments[gid][occ]
                if labels_for_seg:
                    values, counts = np.unique(labels_for_seg, return_counts=True)
                    lbl = values[np.argmax(counts)]
            seg_labels.append((s_idx, e_idx, gid, lbl))

        time = np.arange(len(emg)) / sr
        fig, ax = plt.subplots(2, 1, figsize=(24, 10), sharex=True)

        ax[0].plot(time, emg[:, 0], color="black", linewidth=0.7)
        ax[0].set_title("EMG (канал 1)")
        ax[0].set_ylabel("Амплитуда")
        ax[0].grid(True, alpha=0.3)

        for (s_idx, e_idx, gid, lbl) in seg_labels:
            color = subset_colors.get(lbl, "#BDBDBD") if gid != 0 else "#9E9E9E"
            ax[1].barh(
                0,
                width=(e_idx - s_idx) / sr,
                left=s_idx / sr,
                height=0.8,
                color=color,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.2,
            )
        ax[1].set_title("Segment breakdown: train / val / test")
        ax[1].set_xlabel("Times (sec)")
        ax[1].set_yticks([])
        ax[1].grid(True, alpha=0.3, axis="x")

        legend_labels = ["train", "val", "test", "REST/none"]
        legend_colors = [subset_colors["train"], subset_colors["val"], subset_colors["test"], "#9E9E9E"]
        patches = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black') for c in legend_colors]
        ax[1].legend(patches, legend_labels, loc="upper right", ncol=4)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Segment split visualization saved: {save_path}")

    def plot_training_curves(self, history: Dict[str, List[float]], filename: str = "training_curves.png"):
        """Learning curves: loss/accuracy (train/val)"""
        self.logger.info("Visualization of learning curves (loss/acc)")
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label="Train loss", color="#2196F3", linewidth=2)
        if not np.isnan(history["val_loss"]).all():
            axes[0].plot(epochs, history["val_loss"], label="Val loss", color="#F44336", linewidth=2)
        axes[0].set_title("Loss curve")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, history["train_acc"], label="Train acc", color="#4CAF50", linewidth=2)
        if not np.isnan(history["val_acc"]).all():
            axes[1].plot(epochs, history["val_acc"], label="Val acc", color="#FF9800", linewidth=2)
        axes[1].set_title("Accuracy curve")
        axes[1].set_xlabel("era")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Learning curves saved: {save_path}")

    def plot_confusion_matrix(self, cm: np.ndarray, class_labels: List[str], normalize: bool = True, filename: str = "confusion_matrix.png"):
        cm_plot = cm.astype(np.float32)
        if normalize:
            cm_sum = cm_plot.sum(axis=1, keepdims=True) + 1e-8
            cm_plot = cm_plot / cm_sum

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.imshow(cm_plot, cmap='Blues', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Confusion Matrix" + ("(normal)" if normalize else ""))
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_labels)

        fmt = ".2f" if normalize else "d"
        thresh = cm_plot.max() / 2.0
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                ax.text(j, i, format(cm_plot[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_plot[i, j] > thresh else "black")

        ax.grid(False)
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Error matrix saved: {save_path}")
        
    def plot_per_class_f1(self, report: Dict, class_labels: List[str], filename: str = "per_class_f1.png"):
        """F1 columns by class"""
        f1s = []
        labels = []
        for idx, name in enumerate(class_labels):
            key = str(idx)
            if key in report:
                f1s.append(report[key].get("f1-score", 0.0))
                labels.append(name)
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.bar(labels, f1s, color="#2196F3", alpha=0.8)
        ax.set_title("F1-score by class")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"F1 saved by classes: {save_path}")

    def plot_roc_ovr(self, y_true: np.ndarray, proba: np.ndarray, class_labels: List[str], filename: str = "roc_ovr.png"):
        """ROC curves of OvR for the multi-class case"""
        self.logger.info("ROC OvR Visualization")
        n_classes = proba.shape[1]
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc["macro"] = auc(all_fpr, mean_tpr)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC={roc_auc[i]:.2f})")
        ax.plot(all_fpr, mean_tpr, color='black', linewidth=2, label=f"Macro-AUC={roc_auc['macro']:.2f}")
        ax.set_title("ROC-curve OvR")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"ROC curves saved: {save_path}")

    def plot_two_gestures_full_canvas(
        self,
        emg: np.ndarray,
        stimulus: np.ndarray,
        config: ProcessingConfig,
        grouped_windows: Dict[int, List[np.ndarray]],
        assignments: Dict[int, List[List[str]]],
        filename: str = "two_gestures_full_canvas.png",
        g1: int = 41, 
        g2: int = 42
    ):
        self.logger.info("Creating a full canvas (limited to two selected gestures)")
        sr = config.sampling_rate
        stim = stimulus.flatten()

        change_idxs = np.where(np.diff(stim, prepend=stim[0]) != 0)[0]
        segments_timeline: List[Tuple[int, int, int]] = []
        for i in range(len(change_idxs)):
            start = change_idxs[i]
            end = change_idxs[i + 1] if i + 1 < len(change_idxs) else len(stim)
            gid = int(stim[start])
            segments_timeline.append((start, end, gid))

        # Define the visualization boundaries: from the first occurrence of g1 to the end of the last g2,
        # including only g1, g2, and the rest period between them. Stop when a gesture appears outside {g1,g2,0}.
        idx_first_g1 = next((i for i, (_, _, gid) in enumerate(segments_timeline) if gid == g1), None)
        if idx_first_g1 is None:
            return

        s_crop = segments_timeline[idx_first_g1][0]
        last_g2_end = None
        end_reached = False
        for i in range(idx_first_g1, len(segments_timeline)):
            s_idx, e_idx, gid = segments_timeline[i]
            if gid in (g1, g2, 0):
                if gid == g2:
                    last_g2_end = e_idx
            else:
                end_reached = True
                break

        if last_g2_end is None:
            for i in range(len(segments_timeline) - 1, -1, -1):
                if segments_timeline[i][2] == g2:
                    last_g2_end = segments_timeline[i][1]
                    break

        if last_g2_end is None:
            return

        e_crop = last_g2_end

        # Filter segments in the selected range and by classes {g1,g2,0} with clipping
        filtered_segments: List[Tuple[int, int, int]] = []
        for (s_idx, e_idx, gid) in segments_timeline:
            if gid not in (g1, g2, 0):
                continue
            if e_idx <= s_crop or s_idx >= e_crop:
                continue
            s_cl = max(s_idx, s_crop)
            e_cl = min(e_idx, e_crop)
            if e_cl > s_cl:
                filtered_segments.append((s_cl, e_cl, gid))

        # Time axis and signal only within the window [s_crop:e_crop)
        t_crop = np.arange(s_crop, e_crop) / sr
        sig_crop = emg[s_crop:e_crop, 0]

        class_colors = {g1: "#2196F3", g2: "#9C27B0", 0: "#9E9E9E"}
        subset_colors = {"train": "#4CAF50", "val": "#FFC107", "test": "#F44336"}

        fig, axes = plt.subplots(4, 1, figsize=(24, 18), sharex=True)
        ax1, ax2, ax3, ax4 = axes

        ax1.plot(t_crop, sig_crop, color="black", linewidth=0.7, label="EMG channel 1")
        first_labels_shown = {g1: False, g2: False, 0: False}
        for (s_idx, e_idx, gid) in filtered_segments:
            label = f"Gesture {gid}" if gid != 0 else "Rest"
            ax1.axvspan(
                s_idx / sr,
                (e_idx - 1) / sr if e_idx > s_idx else e_idx / sr,
                color=class_colors[gid],
                alpha=0.15,
                label=label if not first_labels_shown[gid] else None,
            )
            first_labels_shown[gid] = True

        ax1.set_title(f"Original signal (limited by gestures {g1} and {g2} + rest)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper right", ncol=3)

        # Timeline of repetitions (bar on the line for g1, g2, rest) within the window
        y_map = {g1: 2, g2: 1, 0: 0}
        for (s_idx, e_idx, gid) in filtered_segments:
            ax2.barh(
                y_map[gid],
                width=(e_idx - s_idx) / sr,
                left=s_idx / sr,
                height=0.6,
                color=class_colors[gid],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.2,
            )
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["Rest", f"Gesture {g2}", f"Gesture {g1}"])
        ax2.set_title("Repetition split (within the selected gestures)")
        ax2.set_ylabel("Class")
        ax2.grid(True, alpha=0.3, axis="x")

        ws = config.window_size
        step = config.window_size - config.window_overlap

        # Calculate the number of repetitions to place vertically in ax3 and ax4 (only on filtered segments)
        reps_count = {g1: 0, g2: 0, 0: 0}
        for (_, _, gid) in filtered_segments:
            reps_count[gid] += 1

        base_map_ax3 = {}
        cur_base = 0
        for gid in [g1, g2, 0]:
            base_map_ax3[gid] = cur_base
            cur_base += reps_count[gid] + 1  

        # Windows within each rep (for g1, g2 and rest) within the window
        rep_idx_map = {g1: 0, g2: 0, 0: 0}
        for (s_idx, e_idx, gid) in filtered_segments:
            seg_len = e_idx - s_idx
            num_windows = (seg_len - ws) // step + 1
            y = base_map_ax3[gid] + rep_idx_map[gid]
            rep_idx_map[gid] += 1
            if num_windows <= 0:
                continue

            for w in range(num_windows):
                w_start = s_idx + w * step
                ax3.barh(
                    y,
                    width=ws / sr,
                    left=w_start / sr,
                    height=0.6,
                    color=class_colors[gid],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.2,
                )

        ax3.set_title("Windows within each repetition (within the selected gestures)")
        ax3.set_ylabel("Repetitions (stack by class)")
        ax3.grid(True, alpha=0.3, axis="x")

        # Windows colored by train/val/test (only g1 and g2) within the window
        base_map_ax4 = {}
        cur_base = 0
        for gid in [g1, g2]:
            base_map_ax4[gid] = cur_base
            cur_base += reps_count[gid] + 1

        rep_idx_map = {g1: 0, g2: 0}
        for (s_idx, e_idx, gid) in filtered_segments:
            if gid not in (g1, g2):
                continue

            seg_len = e_idx - s_idx
            num_windows = (seg_len - ws) // step + 1
            occ = rep_idx_map[gid]
            rep_idx_map[gid] += 1
            if num_windows <= 0:
                continue

            labels_for_seg: List[str] = []
            if gid in assignments and occ < len(assignments[gid]):
                labels_for_seg = assignments[gid][occ]
            use_n = min(num_windows, len(labels_for_seg)) if labels_for_seg else 0

            y = base_map_ax4[gid] + occ
            for w in range(num_windows):
                w_start = s_idx + w * step
                ax4.barh(
                    y,
                    width=ws / sr,
                    left=w_start / sr,
                    height=0.6,
                    color="#BDBDBD",
                    alpha=0.3,
                    edgecolor="black",
                    linewidth=0.15,
                    zorder=1,
                )
            for w in range(use_n):
                w_start = s_idx + w * step
                lbl = labels_for_seg[w]
                color = subset_colors.get(lbl, "#9E9E9E")
                ax4.barh(
                    y,
                    width=ws / sr,
                    left=w_start / sr,
                    height=0.6,
                    color=color,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.2,
                    zorder=2,
                )

        ax4.set_title("Windows within each iteration, split into train/val/test (selected gestures only)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Iterations (stack by class)")
        ax4.grid(True, alpha=0.3, axis="x")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim([s_crop / sr, e_crop / sr])

        legend_patches = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.9, edgecolor='black') 
                          for c in subset_colors.values()]
        ax4.legend(legend_patches, ['train', 'val', 'test'], loc='upper right', ncol=3)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Canvas (limited to two gestures) saved: {save_path}")

    def plot_split_overview_for_two_gestures(
        self,
        emg: np.ndarray,
        stimulus: np.ndarray,
        config: ProcessingConfig,
        assignments: Dict[int, List[List[str]]],
        filename: str = "split_overview.png",
    ):
        """
        Visualization: We show the first signal channel for two consecutive NON-REST gestures 
        there may be a REST gesture between them), divide it into windows, and mark which windows are in train/val/test.
        """

        sr = config.sampling_rate
        stim = stimulus.flatten()

        change_idxs = np.where(np.diff(stim, prepend=stim[0]) != 0)[0]
        segments_timeline = [] 
        for i in range(len(change_idxs)):
            start = change_idxs[i]
            end = change_idxs[i + 1] if i + 1 < len(change_idxs) else len(stim)
            gid = int(stim[start])
            segments_timeline.append((start, end, gid))

        non_rest_indices = [i for i, (_, _, gid) in enumerate(segments_timeline) if gid != 0]
        if len(non_rest_indices) < 2:
            self.logger.warning("Not enough non-REST segments to render (need >= 2)")
            return

        i1 = non_rest_indices[0]
        i2 = non_rest_indices[1]
        s1, e1, g1 = segments_timeline[i1]
        s2, e2, g2 = segments_timeline[i2]

        def occurrence_index(idx_in_timeline: int) -> int:
            gid_target = segments_timeline[idx_in_timeline][2]
            count = 0
            for j in range(idx_in_timeline):
                if segments_timeline[j][2] == gid_target and gid_target != 0:
                    count += 1
            return count

        occ1 = occurrence_index(i1)
        occ2 = occurrence_index(i2)

        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
        start_plot = s1
        end_plot = e2
        t = np.arange(start_plot, end_plot) / sr
        sig = emg[start_plot:end_plot, 0]
        ax.plot(t, sig, color='black', linewidth=0.8, label='EMG channel 1')

        subset_colors = {
            "train": "#4CAF50",
            "val": "#FFC107",
            "test": "#F44336",
        }

        def draw_segment_windows(seg_start: int, seg_end: int, gid: int, occ_idx: int):
            ws = config.window_size
            step = config.window_size - config.window_overlap
            seg_len = seg_end - seg_start
            num_windows = (seg_len - ws) // step + 1
            if num_windows <= 0:
                return

            labels = assignments.get(gid, [])
            if occ_idx >= len(labels):
                self.logger.warning(f"No assignments for gesture {gid}, repeating {occ_idx}")
                return
            labels_seg = labels[occ_idx]
            if len(labels_seg) != num_windows:
                self.logger.warning(
                    f"Mismatch between the number of windows and the assignment for gesture {gid}, duplicate {occ_idx}: "
                    f"{num_windows} vs {len(labels_seg)}. Trimming to the minimum."
                )
            use_n = min(num_windows, len(labels_seg))

            ymin, ymax = np.min(sig), np.max(sig)
            height = ymax - ymin
            if height == 0:
                height = 1.0
            rect_y = ymin
            for w in range(use_n):
                w_start = seg_start + w * step
                w_end = w_start + ws
                left_t = max((w_start - start_plot) / sr, t[0])
                width_t = (ws) / sr
                lbl = labels_seg[w]
                color = subset_colors.get(lbl, "#9E9E9E")
                rect = Rectangle(
                    (left_t, rect_y),
                    width_t,
                    height,
                    facecolor=color,
                    alpha=0.18,
                    edgecolor='none',
                    zorder=1,
                )
                ax.add_patch(rect)

        draw_segment_windows(s1, e1, g1, occ1)
        draw_segment_windows(s2, e2, g2, occ2)

        ax.axvline(x=(s1 - start_plot)/sr, color='blue', linestyle='--', alpha=0.6)
        ax.axvline(x=(e1 - start_plot)/sr, color='blue', linestyle='--', alpha=0.6)
        ax.axvline(x=(s2 - start_plot)/sr, color='purple', linestyle='--', alpha=0.6)
        ax.axvline(x=(e2 - start_plot)/sr, color='purple', linestyle='--', alpha=0.6)

        ax.set_title(f'Subsampling: gesture {g1} (blue) and gesture {g2} (purple)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

        legend_patches = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.3, edgecolor='none') 
                          for c in subset_colors.values()]
        ax.legend(legend_patches, ['train', 'val', 'test'], loc='upper right')

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Final split visualization saved: {save_path}")
        
    def plot_signal_overview(self, emg: np.ndarray, stimulus: np.ndarray, filename: str = "signal_overview.png"):
        
        fig, axes = plt.subplots(3, 1, figsize=(20, 12))
        
        time = np.arange(len(emg)) / 2000 
        for i in range(min(4, emg.shape[1])):
            axes[0].plot(time, emg[:, i], label=f'Channel {i+1}', alpha=0.7)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('EMG Signal (first 4 channels)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Stimulus
        axes[1].plot(time, stimulus.flatten(), color='red', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Gesture ID')
        axes[1].set_title('Stimulus (gesture labels)')
        axes[1].grid(True)
        
        ax_main = axes[2]
        ax_stimulus = ax_main.twinx()
        
        for i in range(min(2, emg.shape[1])):
            ax_main.plot(time, emg[:, i], label=f'EMG Ch{i+1}', alpha=0.7)
        ax_stimulus.plot(time, stimulus.flatten(), color='red', linewidth=2, label='Stimulus', alpha=0.5)
        
        ax_main.set_xlabel('Time (s)')
        ax_main.set_ylabel('EMG Amplitude')
        ax_stimulus.set_ylabel('Gesture ID')
        ax_main.set_title('Combined View: EMG + Stimulus')
        ax_main.legend(loc='upper left')
        ax_stimulus.legend(loc='upper right')
        ax_main.grid(True)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Overview visualization saved: {save_path}")
    
    def plot_gesture_segments(self, segments: Dict[int, List[np.ndarray]], filename: str = "gesture_segments.png"):
        
        num_gestures = len(segments)
        fig, axes = plt.subplots(num_gestures, 1, figsize=(20, 4 * num_gestures))
        
        if num_gestures == 1:
            axes = [axes]
        
        for idx, (gesture_id, gesture_segs) in enumerate(sorted(segments.items())):
            ax = axes[idx]
            offset = 0
            for seg_idx, segment in enumerate(gesture_segs):
                time = (np.arange(len(segment)) + offset) / 2000
                ax.plot(time, segment[:, 0], alpha=0.7)
                if seg_idx < len(gesture_segs) - 1:
                    ax.axvline(x=time[-1], color='red', linestyle='--', alpha=0.5)
                
                offset += len(segment)
            label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
            ax.set_title(f'{label} - {len(gesture_segs)} segments (channel 1)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Segment visualization saved: {save_path}")
    
    def plot_windows(self, windows_dict: Dict[int, np.ndarray], filename: str = "windows_visualization.png"):
        
        num_gestures = len(windows_dict)
        fig, axes = plt.subplots(num_gestures, 2, figsize=(20, 5 * num_gestures))
        
        if num_gestures == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (gesture_id, windows) in enumerate(sorted(windows_dict.items())):
            num_examples = min(5, len(windows))
            ax_left = axes[idx, 0]
            for i in range(num_examples):
                time = np.arange(windows[i].shape[0]) / 2000 * 1000 
                ax_left.plot(time, windows[i, :, 0], alpha=0.7, label=f'Окно {i+1}')
            
            label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
            ax_left.set_title(f'{label} - Window Examples (Channel 1)')
            ax_left.set_xlabel('Time (ms)')
            ax_left.set_ylabel('Amplitude')
            ax_left.legend()
            ax_left.grid(True)
            
            ax_right = axes[idx, 1]
            max_windows_to_show = 100
            windows_subset = windows[:max_windows_to_show, :, 0].T
            
            im = ax_right.imshow(windows_subset, aspect='auto', cmap='viridis', interpolation='nearest')
            ax_right.set_title(f'{label} - Window Heatmap (Channel 1)')
            ax_right.set_xlabel('Window Number')
            ax_right.set_ylabel('Samples in Window')
            plt.colorbar(im, ax=ax_right)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Window visualization saved: {save_path}")
    
    def plot_statistics(self, windows_dict: Dict[int, np.ndarray], filename: str = "statistics.png"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        gesture_ids = sorted(windows_dict.keys())
        window_counts = [len(windows_dict[gid]) for gid in gesture_ids]
        
        axes[0, 0].bar(gesture_ids, window_counts, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Gesture ID')
        axes[0, 0].set_ylabel('Number of windows')
        axes[0, 0].set_title('Windows distribution by gestures')
        axes[0, 0].grid(True, alpha=0.3)
        
        mean_amplitudes = []
        for gid in gesture_ids:
            mean_amp = np.mean(np.abs(windows_dict[gid]))
            mean_amplitudes.append(mean_amp)
        
        axes[0, 1].bar(gesture_ids, mean_amplitudes, color='coral', alpha=0.7)
        axes[0, 1].set_xlabel('Gesture ID')
        axes[0, 1].set_ylabel('Average Amplitude')
        axes[0, 1].set_title('Average Signal Amplitude by Gesture')
        axes[0, 1].grid(True, alpha=0.3)
        
        first_gesture_id = gesture_ids[0]
        first_gesture_windows = windows_dict[first_gesture_id]
        channel_energy = np.mean(first_gesture_windows**2, axis=(0, 1))
        
        axes[1, 0].bar(range(len(channel_energy)), channel_energy, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xlabel('Channel Number')
        axes[1, 0].set_ylabel('Average Energy')
        axes[1, 0].set_title(f'Energy by Channel (Gesture {first_gesture_id})')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].axis('off')

        table_data = []
        table_data.append(['Gesture ID', 'Windows', 'Channels', 'Samples/window'])
        for gid in gesture_ids:
            windows = windows_dict[gid]
            table_data.append([
                str(gid),
                str(len(windows)),
                str(windows.shape[2]),
                str(windows.shape[1])
            ])
        
        table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                                  colWidths=[0.2, 0.2, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('Summary Statistics', pad=20, fontsize=14, fontweight='bold')      
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Statistics visualization saved: {save_path}")
    
    def plot_windows_timeline(self, windows_dict: Dict[int, np.ndarray], segments: Dict[int, List[np.ndarray]], 
                              emg: np.ndarray, stimulus: np.ndarray, config: ProcessingConfig,
                              filename: str = "windows_timeline.png"):
        fig, axes = plt.subplots(4, 1, figsize=(24, 16))
        
        time_full = np.arange(len(emg)) / 2000
        ax1 = axes[0]
        ax1.plot(time_full, emg[:, 0], alpha=0.5, color='gray', linewidth=0.5, label='EMG channel 1')
        
        stimulus_flat = stimulus.flatten()
        unique_gestures = np.unique(stimulus_flat)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gestures)))
        gesture_colors = {int(g): colors[i] for i, g in enumerate(sorted(unique_gestures))}
        
        current_gesture = stimulus_flat[0]
        start_idx = 0
        
        for i in range(1, len(stimulus_flat)):
            if stimulus_flat[i] != current_gesture:
                end_idx = i
                gesture_id = int(current_gesture)
                label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
                ax1.axvspan(time_full[start_idx], time_full[end_idx], 
                           alpha=0.3, color=gesture_colors[gesture_id], label=label if start_idx == 0 else "")
                current_gesture = stimulus_flat[i]
                start_idx = i
        
        gesture_id = int(current_gesture)
        label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
        ax1.axvspan(time_full[start_idx], time_full[-1], 
                   alpha=0.3, color=gesture_colors[gesture_id], label=label if start_idx == 0 else "")
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original signal with gesture labeling')
        ax1.grid(True, alpha=0.3)
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=5)
        ax2 = axes[1]
        
        step = config.window_size - config.window_overlap
        
        window_positions = []  # (start_time, end_time, gesture_id)
        
        stimulus_flat = stimulus.flatten()
        gesture_changes = np.where(np.diff(stimulus_flat, prepend=stimulus_flat[0]) != 0)[0]
        
        for i in range(len(gesture_changes)):
            start_idx = gesture_changes[i]
            end_idx = gesture_changes[i + 1] if i + 1 < len(gesture_changes) else len(stimulus_flat)
            gesture_id = int(stimulus_flat[start_idx])
            
            segment = emg[start_idx:end_idx]
            num_windows = (segment.shape[0] - config.window_size) // step + 1
            
            if num_windows > 0:
                for w in range(num_windows):
                    window_start_sample = start_idx + w * step
                    window_end_sample = window_start_sample + config.window_size
                    
                    window_start_time = window_start_sample / 2000
                    window_end_time = window_end_sample / 2000
                    
                    window_positions.append((window_start_time, window_end_time, gesture_id))
        
        for start_t, end_t, gesture_id in window_positions:
            ax2.barh(0, end_t - start_t, left=start_t, height=0.8, 
                    color=gesture_colors[gesture_id], alpha=0.7, edgecolor='black', linewidth=0.2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Windows')
        ax2.set_title(f'Time distribution of windows (total {len(window_positions)} windows)')
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, axis='x')
        
        ax3 = axes[2]
        
        time_bins = np.arange(0, time_full[-1], 1.0)
        window_count_per_bin = {g: np.zeros(len(time_bins) - 1) for g in unique_gestures}
        
        for start_t, end_t, gesture_id in window_positions:
            bin_idx = np.digitize(start_t, time_bins) - 1
            if 0 <= bin_idx < len(time_bins) - 1:
                window_count_per_bin[gesture_id][bin_idx] += 1
        
        bottom = np.zeros(len(time_bins) - 1)
        for gesture_id in sorted(unique_gestures):
            label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
            ax3.bar(time_bins[:-1], window_count_per_bin[gesture_id], width=1.0, 
                   bottom=bottom, label=label, color=gesture_colors[gesture_id], alpha=0.7)
            bottom += window_count_per_bin[gesture_id]
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Number of windows')
        ax3.set_title('Number of windows per second')
        ax3.legend(loc='upper right', ncol=5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = axes[3]
        ax4.axis('off')
        
        table_data = [['Class', 'Segments', 'Windows', 'Windows/Segment', '% of all Windows']]
        total_windows = sum(len(w) for w in windows_dict.values())
        
        for gesture_id in sorted(windows_dict.keys()):
            label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
            num_segments = len(segments[gesture_id])
            num_windows = len(windows_dict[gesture_id])
            windows_per_segment = num_windows / num_segments if num_segments > 0 else 0
            percentage = (num_windows / total_windows * 100) if total_windows > 0 else 0
            
            table_data.append([
                label,
                str(num_segments),
                str(num_windows),
                f"{windows_per_segment:.1f}",
                f"{percentage:.1f}%"
            ])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for row_idx in range(1, len(table_data)):
            gesture_id = sorted(windows_dict.keys())[row_idx - 1]
            for col_idx in range(5):
                table[(row_idx, col_idx)].set_facecolor(gesture_colors[gesture_id])
                table[(row_idx, col_idx)].set_alpha(0.3)
        
        ax4.set_title('Window Distribution Statistics', pad=20, fontsize=14, fontweight='bold')     
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Temporary window rendering saved: {save_path}")
