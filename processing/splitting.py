import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from typing import Dict, List, Tuple
import logging
from config.base import SplitConfig

class DatasetSplitter:
    """Splitting windows/segments into train/val/test"""
    def __init__(self, split_config: SplitConfig, logger: logging.Logger):
        self.cfg = split_config
        self.logger = logger
        self.cfg.validate()
        self.rng = np.random.RandomState(self.cfg.seed)

    def split_grouped_windows(
        self, grouped_windows: Dict[int, List[np.ndarray]]
    ) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, List[List[str]]]]:
        """
        Args:
            grouped_windows: {gesture_id: [windows_seg0, windows_seg1, ...]},
                             where windows_seg.shape = (num_windows, window_size, channels)

        Returns:
            splits: {
                "train": {gid: np.ndarray(...), ...},
                "val":   {gid: np.ndarray(...), ...},
                "test":  {gid: np.ndarray(...), ...},
            }
            assignments: {gid: [[labels_for_seg0], [labels_for_seg1], ...]}
                         labels ∈ {"train","val","test"} for each segment window
        """
        if self.cfg.mode == "by_windows":
            return self._split_by_windows(grouped_windows)
        else:
            return self._split_by_segments(grouped_windows)

    def _split_by_windows(
        self, grouped_windows: Dict[int, List[np.ndarray]]
    ) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, List[List[str]]]]:
        """Initial logic: we sequentially cut windows within a segment"""
        self.logger.info("Partitioning: by_windows (sequential division of windows within repetitions)")
        splits = {"train": {}, "val": {}, "test": {}}
        # temporary accumulators for concatenation
        def init_accum():
            return {
                "train": {},
                "val":   {},
                "test":  {},
            }
        accum = init_accum()
        assignments: Dict[int, List[List[str]]] = {}

        for gid, seg_windows_list in grouped_windows.items():
            if gid == 0 and not self.cfg.include_rest_in_splits:
                # Ignore REST when generating subselects
                continue

            assignments[gid] = []
            for split_name in ["train", "val", "test"]:
                accum[split_name].setdefault(gid, [])

            for seg_idx, seg_windows in enumerate(seg_windows_list):
                n = len(seg_windows)
                if n == 0:
                    assignments[gid].append([])
                    continue

                n_train = int(np.floor(n * self.cfg.train_ratio))
                n_val   = int(np.floor(n * self.cfg.val_ratio))
                n_test  = n - n_train - n_val

                labels = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
                assignments[gid].append(labels)

                if n_train > 0:
                    accum["train"][gid].append(seg_windows[:n_train])
                if n_val > 0:
                    accum["val"][gid].append(seg_windows[n_train:n_train + n_val])
                if n_test > 0:
                    accum["test"][gid].append(seg_windows[n_train + n_val:])

                self.logger.info(
                    f"[by_windows] Gesture {gid}, repeat {seg_idx}: window={n} -> "
                    f"train={n_train}, val={n_val}, test={n_test}"
                )

        for split_name in ["train", "val", "test"]:
            for gid, parts in accum[split_name].items():
                if parts:
                    splits[split_name][gid] = np.concatenate(parts, axis=0)
                else:
                    splits[split_name][gid] = np.empty((0,), dtype=np.float32)

        for split_name in ["train", "val", "test"]:
            total = sum(len(arr) for arr in splits[split_name].values()
                        if isinstance(arr, np.ndarray) and arr.ndim == 3)
            self.logger.info(f"{split_name.upper()}: всего окон = {total}")

        return splits, assignments

    def _split_by_segments(
        self, grouped_windows: Dict[int, List[np.ndarray]]
    ) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, List[List[str]]]]:
        """New logic: entire segments (repetitions) in train/val/test"""
        self.logger.info("Split: by_segments (whole repetitions in one split)")
        splits = {"train": {}, "val": {}, "test": {}}
        assignments: Dict[int, List[List[str]]] = {}

        for gid, seg_list in grouped_windows.items():
            if gid == 0 and not self.cfg.include_rest_in_splits:
                continue

            n_seg = len(seg_list)
            idxs = np.arange(n_seg)
            if self.cfg.shuffle_segments:
                self.rng.shuffle(idxs)

            n_train = int(np.floor(n_seg * self.cfg.train_ratio))
            n_val   = int(np.floor(n_seg * self.cfg.val_ratio))
            n_test  = n_seg - n_train - n_val

            seg_train = idxs[:n_train]
            seg_val   = idxs[n_train:n_train+n_val]
            seg_test  = idxs[n_train+n_val:]

            # Concatenate entire windows into segments
            splits["train"][gid] = (np.concatenate([seg_list[i] for i in seg_train], axis=0)
                                    if len(seg_train) > 0 else np.empty((0,), dtype=np.float32))
            splits["val"][gid]   = (np.concatenate([seg_list[i] for i in seg_val], axis=0)
                                    if len(seg_val) > 0 else np.empty((0,), dtype=np.float32))
            splits["test"][gid]  = (np.concatenate([seg_list[i] for i in seg_test], axis=0)
                                    if len(seg_test) > 0 else np.empty((0,), dtype=np.float32))

            assignments[gid] = []
            for i, seg in enumerate(seg_list):
                if i in seg_train:
                    labels = ["train"] * len(seg)
                elif i in seg_val:
                    labels = ["val"] * len(seg)
                elif i in seg_test:
                    labels = ["test"] * len(seg)
                else:
                    labels = []
                assignments[gid].append(labels)

            self.logger.info(
                f"[by_segments] Gesture {gid}: segments ={n_seg} -> "
                f"train={len(seg_train)}, val={len(seg_val)}, test={len(seg_test)}"
            )

        for split_name in ["train", "val", "test"]:
            total = sum(len(arr) for arr in splits[split_name].values()
                        if isinstance(arr, np.ndarray) and arr.ndim == 3)
            self.logger.info(f"{split_name.upper()}: all window = {total}")

        return splits, assignments