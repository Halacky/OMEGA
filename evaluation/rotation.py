import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import torch
from config.base import RotationConfig
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from visualization.rotation import RotationVisualizer

def build_rotation_permutation(
    C: int,
    shift: int,
    bracelet_size: Optional[int] = None,
    channel_order: Optional[List[int]] = None,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Returns permutation p of shape (C,) such that new_X[..., i] = orig_X[..., p[i]]
    That is: we remap source channels to the target model input index i after a 'shift' of sensor positions.

    - C: number of input channels to the model
    - shift: rotation on bracelet positions (negative == clockwise if positions increase around bracelet)
    - bracelet_size: total number of positions on the bracelet (default C)
    - channel_order: for each input channel i (0..C-1), channel_order[i] = bracelet position (0..bracelet_size-1)
    """
    if bracelet_size is None:
        bracelet_size = C
    if channel_order is None:
        channel_order = list(range(C))

    if len(channel_order) != C:
        raise ValueError("channel_order length must equal number of channels C")
    if len(set(channel_order)) != len(channel_order):
        raise ValueError("channel_order must be unique")
    if not all(0 <= p < bracelet_size for p in channel_order):
        raise ValueError("channel_order positions must be within [0, bracelet_size-1]")

    # Map: position -> input-channel index
    pos_to_ch: Dict[int, int] = {pos: ch for ch, pos in enumerate(channel_order)}

    perm = np.zeros(C, dtype=np.int64)
    for i in range(C):
        pos_i = channel_order[i]             # target position that model's i-th channel expects
        src_pos = (pos_i + shift) % bracelet_size  # where the physical sensor data comes from after rotation
        if src_pos not in pos_to_ch:
            # If subset of positions is used and src_pos is not present, default to original channel (or zero-fill externally)
            if logger:
                logger.warning(f"Rotation maps to missing position {src_pos}; keeping original channel index")
            perm[i] = i
        else:
            perm[i] = pos_to_ch[src_pos]
    return perm

def apply_channel_permutation(X: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Apply permutation on channel axis for (N,T,C): new_X[..., i] = X[..., perm[i]]
    """
    assert X.ndim == 3, "Expected X with shape (N,T,C)"
    C = X.shape[2]
    if perm.shape[0] != C:
        raise ValueError(f"Permutation length {perm.shape[0]} != channels {C}")
    return X[:, :, perm]

class RotationExperiment:
    def __init__(self,
                 trainer: WindowClassifierTrainer,
                 base_visualizer: Visualizer,
                 rot_visualizer: RotationVisualizer,
                 logger: logging.Logger,
                 rot_cfg: RotationConfig):
        self.trainer = trainer
        self.base_viz = base_visualizer
        self.rot_viz = rot_visualizer
        self.logger = logger
        self.cfg = rot_cfg

    def _concat_xy(self, dct: Dict[int, np.ndarray], class_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for i, gid in enumerate(class_ids):
            if gid in dct:
                arr = dct[gid]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                    X_list.append(arr)
                    y_list.append(np.full((len(arr),), i, dtype=np.int64))
        if len(X_list) == 0:
            return np.empty((0,)), np.empty((0,), dtype=np.int64)
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def evaluate_full_test_with_rotations(self,
                                          splits: Dict[str, Dict[int, np.ndarray]]) -> Dict[int, Dict]:
        """
        Evaluate baseline (rotation 0) and rotated test sets.
        Returns: rotation_to_metrics dict, each entry contains standard metrics.
        """
        assert self.trainer.class_ids is not None, "Trainer class_ids missing"
        class_ids = self.trainer.class_ids
        X_test, y_test = self._concat_xy(splits["test"], class_ids)
        if len(X_test) == 0:
            self.logger.warning("Empty test split. Skipping rotation experiment on full test.")
            return {}

        C = X_test.shape[2]
        bracelet_size = self.cfg.bracelet_size or C
        ch_order = self.cfg.channel_order or list(range(C))

        rotation_to_metrics: Dict[int, Dict] = {}
        for r in self.cfg.rotations:
            perm = build_rotation_permutation(C=C, shift=r, bracelet_size=bracelet_size,
                                              channel_order=ch_order, logger=self.logger)
            X_rot = apply_channel_permutation(X_test, perm)
            m = self.trainer.evaluate_numpy(X_rot, y_test, split_name=f"test_rot{r}", visualize=False)
            rotation_to_metrics[r] = {k: v for k, v in m.items() if k not in ("logits",)}
            self.logger.info(f"[Rotation {r}] Test accuracy={m['accuracy']:.4f}, f1_macro={m['f1_macro']:.4f}")
        return rotation_to_metrics

    def evaluate_single_test_segment_with_rotations(
        self,
        grouped_windows: Dict[int, List[np.ndarray]],
        assignments: Dict[int, List[List[str]]],
        gesture_id: Optional[int] = None,
        occurrence: Optional[int] = None
    ) -> Tuple[Dict[int, np.ndarray], int, int]:
        """
        Picks a test segment and returns per-rotation probabilities:
        - probs_by_rot[r] -> (W, C) probabilities for windows in that segment
        Also returns chosen gesture_id and occurrence index.
        """
        # auto-pick first test segment if not provided
        if gesture_id is None or occurrence is None:
            found = False
            for gid, seg_labels in assignments.items():
                for occ, labels in enumerate(seg_labels):
                    if labels and all(lbl == "test" for lbl in labels):
                        gesture_id, occurrence = gid, occ
                        found = True
                        break
                if found:
                    break
            if not found:
                self.logger.warning("No pure 'test' segment found in assignments; picking any segment that contains 'test' labels")
                for gid, seg_labels in assignments.items():
                    for occ, labels in enumerate(seg_labels):
                        if labels and ("test" in labels):
                            gesture_id, occurrence = gid, occ
                            found = True
                            break
                    if found:
                        break
        assert gesture_id is not None and occurrence is not None, "Could not determine a test segment for per-segment evaluation"

        seg_windows = grouped_windows.get(gesture_id, [])[occurrence]
        if not isinstance(seg_windows, np.ndarray) or seg_windows.ndim != 3 or len(seg_windows) == 0:
            self.logger.warning("Selected segment has no windows; skipping per-segment rotation eval")
            return {}, gesture_id, occurrence

        C = seg_windows.shape[2]
        bracelet_size = self.cfg.bracelet_size or C
        ch_order = self.cfg.channel_order or list(range(C))

        # y for this segment are the same class index per window
        assert self.trainer.class_ids is not None, "Trainer class_ids missing"
        class_ids = self.trainer.class_ids
        if gesture_id not in class_ids:
            self.logger.warning(f"Gesture {gesture_id} not in trained class_ids; per-segment eval may be inconsistent")
        cls_index = class_ids.index(gesture_id) if gesture_id in class_ids else 0
        y_seg = np.full((len(seg_windows),), cls_index, dtype=np.int64)

        probs_by_rot: Dict[int, np.ndarray] = {}
        for r in self.cfg.rotations:
            perm = build_rotation_permutation(C=C, shift=r, bracelet_size=bracelet_size,
                                              channel_order=ch_order, logger=self.logger)
            X_rot = apply_channel_permutation(seg_windows, perm)
            out = self.trainer.evaluate_numpy(X_rot, y_seg, split_name=f"seg_gid{gesture_id}_occ{occurrence}_rot{r}",
                                              visualize=False)
            logits = out["logits"]
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            probs_by_rot[r] = probs
        return probs_by_rot, gesture_id, occurrence