import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional


class RotationVisualizer:
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

    def plot_accuracy_vs_rotation(self, rotation_to_metrics: Dict[int, Dict], filename: str = "acc_vs_rotation.png"):
        rots = sorted(rotation_to_metrics.keys())
        acc = [rotation_to_metrics[r]["accuracy"] for r in rots]
        f1m = [rotation_to_metrics[r]["f1_macro"] for r in rots]

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax1.plot(rots, acc, marker="o", label="Accuracy", color="#4CAF50", linewidth=2)
        ax1.plot(rots, f1m, marker="s", label="F1-macro", color="#2196F3", linewidth=2)
        ax1.axvline(0, color="black", linestyle="--", alpha=0.4)
        ax1.set_xlabel("Rotation (sensor shift)")
        ax1.set_ylabel("Score")
        ax1.set_title("Performance vs. Sensor Rotation")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved: {save_path}")

    def plot_cm_grid_for_rotations(self, rotation_to_metrics: Dict[int, Dict], class_labels: List[str],
                                   rotations_to_show: Optional[List[int]] = None,
                                   filename: str = "cm_grid_rotations.png", normalize: bool = True):
        if rotations_to_show is None:
            rotations_to_show = sorted(rotation_to_metrics.keys())
        k = len(rotations_to_show)
        cols = min(4, k)
        rows = int(np.ceil(k / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])

        for idx, r in enumerate(rotations_to_show):
            cm = np.array(rotation_to_metrics[r]["confusion_matrix"], dtype=np.float32)
            if normalize:
                cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
            i = idx // cols
            j = idx % cols
            ax = axes[i, j]
            im = ax.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0.0, vmax=1.0 if normalize else None)
            ax.set_title(f"Rotation {r}")
            ax.set_xticks(np.arange(len(class_labels))); ax.set_yticks(np.arange(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right'); ax.set_yticklabels(class_labels)
            fmt = ".2f" if normalize else "d"
            thresh = cm.max() / 2.0
            for a in range(cm.shape[0]):
                for b in range(cm.shape[1]):
                    ax.text(b, a, format(cm[a, b], fmt),
                            ha="center", va="center",
                            color="white" if cm[a, b] > thresh else "black")
        # colorbar
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax)
        for idx in range(k, rows*cols):
            i = idx // cols; j = idx % cols
            axes[i, j].axis('off')
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved: {save_path}")

    def plot_trueclass_prob_heatmap_for_segment(self, probs_by_rotation: Dict[int, np.ndarray],
                                                true_class_index: int,
                                                filename: str = "segment_trueclass_prob_heatmap.png"):
        """
        probs_by_rotation[r]: (N_windows, num_classes) probabilities for the segment windows
        Plots heatmap of P(true_class) across rotations vs window index.
        """
        rots = sorted(probs_by_rotation.keys())
        rows = len(rots)
        # Build matrix [rows=rotations, cols=windows]
        max_W = max(v.shape[0] for v in probs_by_rotation.values())
        mat = np.zeros((rows, max_W), dtype=np.float32)
        for i, r in enumerate(rots):
            p = probs_by_rotation[r]
            pt = p[:, true_class_index]  # (W,)
            mat[i, :len(pt)] = pt

        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        im = ax.imshow(mat, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(rows))
        ax.set_yticklabels([str(r) for r in rots])
        ax.set_xlabel("Window index in segment")
        ax.set_ylabel("Rotation")
        ax.set_title("True-class probability across rotations and windows")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved: {save_path}")


