# /home/kirill/projects_2/folium/NIR/OMEGA/evaluation/cross_subject_rotation.py

import numpy as np
from typing import Dict, List, Optional
import logging

from evaluation.rotation import build_rotation_permutation, apply_channel_permutation
from training.trainer import WindowClassifierTrainer
from visualization.rotation import RotationVisualizer
from config.base import RotationConfig


class CrossSubjectRotationExperiment:
    """
    Итеративный поворот сигналов ТОЛЬКО тестируемого субъекта
    в контексте уже выполненного cross-subject эксперимента.

    Предполагается, что:
    - модель уже обучена (trainer.model не None)
    - есть common_gestures и trainer.class_ids, соответствующие им
    - есть исходные grouped_windows для тестового субъекта (без поворота)
    """

    def __init__(
        self,
        trainer: WindowClassifierTrainer,
        logger: logging.Logger,
        rot_cfg: RotationConfig,
        rot_visualizer: RotationVisualizer,
    ):
        self.trainer = trainer
        self.logger = logger
        self.cfg = rot_cfg
        self.rot_viz = rot_visualizer

    def _build_test_xy_from_grouped(
        self,
        grouped_windows_test: Dict[int, List[np.ndarray]],
        common_gestures: List[int],
    ):
        """
        Превращает grouped_windows тестового субъекта в X_test, y_test
        по тем же классам, что использовались при обучении (trainer.class_ids).
        """
        assert self.trainer.class_ids is not None, "Trainer class_ids missing"
        class_ids = self.trainer.class_ids

        X_list, y_list = [], []
        for gid in sorted(common_gestures):
            if gid not in grouped_windows_test:
                continue
            if gid not in class_ids:
                # класс не был в обучении (хотя это маловероятно при корректной подготовке common_gestures)
                self.logger.warning(f"Gesture {gid} is not in trained class_ids; skipping for rotation test")
                continue

            reps = grouped_windows_test[gid]
            if not reps:
                continue

            X_g = np.concatenate(reps, axis=0)  # (N_g, T, C)
            cls_idx = class_ids.index(gid)
            y_g = np.full((len(X_g),), cls_idx, dtype=np.int64)

            X_list.append(X_g)
            y_list.append(y_g)

        if not X_list:
            self.logger.warning("No valid windows for test subject to run rotation experiment")
            return np.empty((0,)), np.empty((0,), dtype=np.int64)

        X_test = np.concatenate(X_list, axis=0)
        y_test = np.concatenate(y_list, axis=0)
        return X_test, y_test

    def run_full_rotation_on_test_subject(
        self,
        grouped_windows_test: Dict[int, List[np.ndarray]],
        common_gestures: List[int],
        experiment_name: str = "cross_subject_rotation_test",
        visualize_per_rotation: bool = False,
    ) -> Dict[int, Dict]:
        """
        Основной метод:
        - Строим X_test, y_test из grouped_windows тестового субъекта.
        - Для каждой ротации из self.cfg.rotations:
            - применяем перестановку каналов
            - оцениваем модель
            - сохраняем метрики (accuracy, f1_macro, cm, report)
        - Опционально можем отрисовывать confusion matrices для отдельных ротаций.

        Возвращает:
            rotation_to_metrics: {rotation: {accuracy, f1_macro, report, confusion_matrix}}
        """
        assert self.trainer.model is not None, "Trainer model is not trained/loaded"
        assert self.trainer.class_ids is not None, "Trainer class_ids missing"

        X_test, y_test = self._build_test_xy_from_grouped(grouped_windows_test, common_gestures)
        if X_test.ndim != 3 or len(X_test) == 0:
            self.logger.warning("Empty X_test for rotation experiment")
            return {}

        C = X_test.shape[2]
        bracelet_size = self.cfg.bracelet_size or C
        ch_order = self.cfg.channel_order or list(range(C))

        rotation_to_metrics: Dict[int, Dict] = {}

        for r in self.cfg.rotations:
            self.logger.info(f"[Rotation {r}] Applying channel permutation and evaluating...")
            perm = build_rotation_permutation(
                C=C,
                shift=r,
                bracelet_size=bracelet_size,
                channel_order=ch_order,
                logger=self.logger,
            )
            X_rot = apply_channel_permutation(X_test, perm)

            # split_name нужен только для имени файлов, если visualize=True
            split_name = f"{experiment_name}_rot{r}"

            out = self.trainer.evaluate_numpy(
                X_rot,
                y_test,
                split_name=split_name,
                visualize=visualize_per_rotation,
            )

            # убираем logits, чтобы не раздувать JSON
            metrics = {k: v for k, v in out.items() if k != "logits"}
            rotation_to_metrics[r] = metrics

            self.logger.info(
                f"[Rotation {r}] Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}"
            )

        # Глобальные визуализации: acc vs rotation + сетка CМ
        try:
            self.rot_viz.plot_accuracy_vs_rotation(
                rotation_to_metrics,
                filename="cross_subject_acc_vs_rotation.png",
            )

            class_labels = [
                ("REST" if gid == 0 else f"Gesture {gid}")
                for gid in self.trainer.class_ids
            ]
            # Выберем несколько характерных ротаций (0 + симметричные)
            show_rots = [r for r in sorted(rotation_to_metrics.keys()) if r in (-3, -2, -1, 0, 1, 2, 3)]
            if not show_rots:
                show_rots = sorted(rotation_to_metrics.keys())[:4]

            self.rot_viz.plot_cm_grid_for_rotations(
                rotation_to_metrics,
                class_labels=class_labels,
                rotations_to_show=show_rots,
                filename="cross_subject_cm_grid_rotations.png",
                normalize=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to create rotation visualizations: {e}")

        return rotation_to_metrics