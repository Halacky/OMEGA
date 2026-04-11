from typing import Dict, List
import numpy as np
import logging

from config.base import RotationConfig
from .rotation import build_rotation_permutation, apply_channel_permutation

def augment_grouped_windows_with_rotations(
    train_grouped: Dict[int, List[np.ndarray]],
    rot_cfg: RotationConfig,
    logger: logging.Logger
) -> Dict[int, List[np.ndarray]]:
    logger.info("Starting rotation-based augmentation of train grouped windows")
    aug_grouped: Dict[int, List[np.ndarray]] = {}
    C = None
    for reps in train_grouped.values():
        for seg in reps:
            if isinstance(seg, np.ndarray) and seg.ndim == 3 and len(seg) > 0:
                C = seg.shape[2]
                break
        if C is not None:
            break
    if C is None:
        logger.warning("No non-empty segments in train_grouped; skipping augmentation.")
        return train_grouped
    bracelet_size = rot_cfg.bracelet_size or C
    ch_order = rot_cfg.channel_order or list(range(C))
    rotation_perms = {}
    for r in rot_cfg.rotations:
        perm = build_rotation_permutation(
            C=C,
            shift=r,
            bracelet_size=bracelet_size,
            channel_order=ch_order,
            logger=logger,
        )
        rotation_perms[r] = perm
        logger.info(f"Built permutation for rotation {r}: {perm}")
    for gid, reps in train_grouped.items():
        aug_grouped[gid] = []
        for seg_idx, seg in enumerate(reps):
            aug_grouped[gid].append(seg)
            if not (isinstance(seg, np.ndarray) and seg.ndim == 3 and len(seg) > 0):
                continue
            for r in rot_cfg.rotations:
                perm = rotation_perms[r]
                seg_rot = apply_channel_permutation(seg, perm)
                aug_grouped[gid].append(seg_rot)
            logger.info(
                f"Gesture {gid}, segment {seg_idx}: original windows={len(seg)} "
                f"+ {len(rot_cfg.rotations)} rotations -> total {len(aug_grouped[gid])} segment entries so far"
            )
    for gid in sorted(aug_grouped.keys()):
        num_reps = len(aug_grouped[gid])
        total_windows = sum(len(rep) for rep in aug_grouped[gid] if isinstance(rep, np.ndarray) and rep.ndim == 3)
        logger.info(f"[Augmented] Gesture {gid}: repetitions={num_reps}, windows={total_windows}")
    return aug_grouped

def create_virtual_rotated_subjects(
    subjects_data: Dict[str, tuple],
    train_subject_ids: List[str],
    rot_cfg: RotationConfig,
    logger: logging.Logger,
) -> Dict[str, tuple]:
    """
    Create virtual subjects by applying channel rotations to grouped windows of train subjects.

    For each subject in `train_subject_ids` and for each rotation in `rot_cfg.rotations`,
    this function creates a new subject entry with id f"{orig_subject}_rot_{r}" and
    rotated grouped_windows.

    The original subjects_data is not modified; a new extended dict is returned.

    This function operates only on the grouped windows; EMG and segments are passed through
    unchanged because they are not used for training in the cross-subject windows pipeline.
    """
    logger.info(
        "Creating virtual rotated subjects from train subjects using rotation configuration: "
        f"rotations={rot_cfg.rotations}, bracelet_size={rot_cfg.bracelet_size}, "
        f"channel_order={rot_cfg.channel_order}"
    )

    extended_data: Dict[str, tuple] = dict(subjects_data)

    # Determine channel count C from any non-empty grouped windows of any train subject
    C = None
    for sid in train_subject_ids:
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]
        for reps in grouped_windows.values():
            for seg in reps:
                if isinstance(seg, np.ndarray) and seg.ndim == 3 and len(seg) > 0:
                    C = seg.shape[2]
                    break
            if C is not None:
                break
        if C is not None:
            break

    if C is None:
        logger.warning(
            "Could not determine channel dimension from train subjects; "
            "no virtual rotated subjects will be created."
        )
        return extended_data

    bracelet_size = rot_cfg.bracelet_size or C
    ch_order = rot_cfg.channel_order or list(range(C))

    # Precompute permutations for all rotations
    rotation_perms = {}
    for r in rot_cfg.rotations:
        perm = build_rotation_permutation(
            C=C,
            shift=r,
            bracelet_size=bracelet_size,
            channel_order=ch_order,
            logger=logger,
        )
        rotation_perms[r] = perm
        logger.info(f"[VirtualSubjects] Built permutation for rotation {r}: {perm}")

    # Create virtual subjects
    for sid in train_subject_ids:
        if sid not in subjects_data:
            logger.warning(f"[VirtualSubjects] Train subject {sid} is not present in subjects_data, skipping.")
            continue

        emg, segments, grouped_windows = subjects_data[sid]

        for r in rot_cfg.rotations:
            virt_sid = f"{sid}_rot_{r}"
            if virt_sid in extended_data:
                logger.warning(f"[VirtualSubjects] Virtual subject {virt_sid} already exists, skipping.")
                continue

            perm = rotation_perms[r]

            # Rotate each segment's windows for each gesture
            rotated_grouped = {}
            for gid, reps in grouped_windows.items():
                rotated_grouped[gid] = []
                for seg_idx, seg in enumerate(reps):
                    if isinstance(seg, np.ndarray) and seg.ndim == 3 and len(seg) > 0:
                        seg_rot = apply_channel_permutation(seg, perm)
                        rotated_grouped[gid].append(seg_rot)
                    else:
                        # Keep empty segments as is
                        rotated_grouped[gid].append(seg)
                logger.info(
                    f"[VirtualSubjects] Subject {sid}, rotation {r}, gesture {gid}: "
                    f"{len(rotated_grouped[gid])} segments"
                )

            # For compatibility, reuse original emg/segments (not used for training in this pipeline)
            extended_data[virt_sid] = (emg, segments, rotated_grouped)
            logger.info(
                f"[VirtualSubjects] Created virtual subject {virt_sid} from {sid} "
                f"with rotation {r}"
            )

    logger.info(
        f"[VirtualSubjects] Total subjects before={len(subjects_data)}, "
        f"after={len(extended_data)}"
    )
    return extended_data