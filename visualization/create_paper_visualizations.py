#!/usr/bin/env python3
"""
create_paper_tables_and_figures.py

Скрипт для статьи:
"A Reproducible Benchmark for Zero-Shot Cross-Subject Hand Gesture Recognition Using sEMG Signals"

Что делает:
- Собирает результаты по всем экспериментам (deep + ML, LOSO по DB2, E1)
- Строит сводные таблицы (CSV + LaTeX):
    * Table 1: все модели
    * Table 2: только deep-модели
    * Table 3: классические ML-модели (SVM, RF, MLP, Hybrid)
    * Class-wise таблица для выбранных экспериментов (по confusion matrix)
- Строит фигуры (по мотивам PaperVisualizationGenerator):
    * Fig 1: распределение LOSO Accuracy и F1-макро (violinplot)
    * Fig 2: inter-subject variability для выбранной модели
    * Fig 3: кривые обучения (если есть истории)
    * Fig 4: средняя confusion matrix для выбранной модели
    * Fig 5: scatter Accuracy vs F1 (по fold'ам)
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# Matplotlib (paper style)
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif"
    ],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3
})

# Цвета для ключевых моделей (можно расширить/менять)
MODEL_COLORS = {
    "simple_cnn": "#2E86AB",
    "attention_cnn": "#A23B72",
    "multiscale_cnn": "#F18F01",
    "bilstm": "#6C757D",
    "bilstm_attention": "#E85D75",
    "bigru": "#8E44AD",
    "tcn": "#16A085",
    "tcn_attn": "#27AE60",
    "cnn_gru_attention": "#D35400",
    "cnn_lstm": "#7E5109",

    # TD deep (exp2) – условно другие цвета
    "td_simple_cnn": "#1F618D",
    "td_multiscale_cnn": "#9C640C",

    # Powerful MLP
    "powerful_mlp": "#34495E",

    # ML models
    "svm_linear": "#C0392B",
    "svm_rbf": "#E74C3C",
    "rf": "#16A085",
    "hybrid_powerful_deep": "#9B59B6",
}

# =============================================================================
# Helper
# =============================================================================
def clean_label(name: str) -> str:
    """
    Делает имя модели более читаемым.
    """
    return (name.replace("_", " ")
                .replace("cnn", "CNN")
                .replace("lstm", "LSTM")
                .replace("gru", "GRU")
                .replace("bilstm", "BiLSTM")
                .replace("bigru", "BiGRU")
                .replace("tcn", "TCN")
                .title())


# =============================================================================
# Experiment Aggregator
# =============================================================================
class ExperimentAggregator:
    """
    Читает артефакты экспериментов и собирает результаты в удобные структуры.
    Поддерживает deep и ML (SVM/RF/MLP/Hybrid), LOSO cross-subject.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.rows = []            # fold-level (per subject) результаты
        self.per_subject_rows = []  # если есть per_subject_analysis в cross_subject_results

    def load_experiment(self, exp_name: str):
        exp_dir = self.base_dir / exp_name
        summary_path = exp_dir / "loso_summary.json"
        if not summary_path.exists():
            print(f"[WARN] Missing loso_summary.json: {exp_name}")
            return

        summary = json.loads(summary_path.read_text())
        model_name = summary["models"][0]
        model_dir = exp_dir / model_name

        for fold_dir in sorted(model_dir.glob("test_DB2_s*")):
            subject = fold_dir.name.replace("test_", "")

            cls_path = fold_dir / "classification_results.json"
            ml_cls_path = fold_dir / "ml_classification_results.json"
            cross_results_path = fold_dir / "cross_subject_results.json"

            ml_mode = False
            cls = None

            if ml_cls_path.exists():
                ml_mode = True
                cls = json.loads(ml_cls_path.read_text())
            elif cls_path.exists():
                cls = json.loads(cls_path.read_text())
            elif cross_results_path.exists():
                # Бывает, что нет отдельных classification_results, есть только cross_subject_results
                cross = json.loads(cross_results_path.read_text())
                if "training" in cross and "test" in cross["training"]:
                    cls = {
                        "class_ids": cross["training"]["class_ids"],
                        "class_names": cross["training"]["class_names"],
                        "test": cross["training"]["test"],
                    }
                elif "cross_subject_test" in cross:
                    cls = {
                        "class_ids": cross["training"]["class_ids"],
                        "class_names": cross["training"]["class_names"],
                        "test": cross["cross_subject_test"],
                    }

            if cls is None:
                print(f"[WARN] No classification results in {fold_dir}")
                continue

            test_metrics = cls["test"]
            acc = test_metrics["accuracy"]
            f1 = test_metrics["f1_macro"]
            cm = np.array(test_metrics["confusion_matrix"])
            class_names = list(cls["class_names"].values())

            # Пытаемся прочитать историю обучения (если это deep)
            history_path = fold_dir / "training_history.json"
            history = None
            if history_path.exists():
                try:
                    history = json.loads(history_path.read_text())
                except Exception:
                    history = None

            self.rows.append({
                "exp_name": exp_name,
                "model_name": model_name,
                "subject": subject,
                "accuracy": acc,
                "f1_macro": f1,
                "confusion_matrix": cm,
                "class_names": class_names,
                "history": history,
                "is_ml": ml_mode,
            })

            # per_subject_analysis (как в примере SVM-RBF)
            if cross_results_path.exists():
                try:
                    cross = json.loads(cross_results_path.read_text())
                    if "per_subject_analysis" in cross:
                        for s_id, s_info in cross["per_subject_analysis"].items():
                            self.per_subject_rows.append({
                                "exp_name": exp_name,
                                "model_name": model_name,
                                "subject": s_id,
                                "accuracy": s_info["accuracy"],
                                "f1_macro": s_info["f1_macro"],
                                "num_samples": s_info.get("num_samples", None),
                                "is_train": s_info.get("is_train", None),
                                "is_test": s_info.get("is_test", None),
                            })
                except Exception:
                    pass

        print(f"✓ Loaded {exp_name}")

    def load_all(self, exp_names):
        for name in exp_names:
            self.load_experiment(name)

    def to_fold_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_per_subject_df(self) -> pd.DataFrame:
        if not self.per_subject_rows:
            return pd.DataFrame()
        return pd.DataFrame(self.per_subject_rows)


# =============================================================================
# PaperVisualizationGenerator (адаптирован для использования с агрегатором)
# =============================================================================
class PaperVisualizationGenerator:

    def __init__(self, output_dir: Path, experiments: dict):
        """
        experiments: dict
            ключ: (exp_name, model_name) или просто строка
            значение: {
                "name": str,
                "model": str,
                "folds": [
                    {
                        "subject": str,
                        "acc": float,
                        "f1": float,
                        "confusion_matrix": np.ndarray,
                        "class_names": list[str],
                        "history": dict | None
                    },
                    ...
                ]
            }
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = experiments

    # -------------------------------------------------------------------------
    def fig_loso_distribution(self):
        """
        Fig 1: распределение LOSO Accuracy и F1-макро (violinplot)
        Увеличена ширина фигуры и угол наклона подписей до 45 градусов
        """
        acc_data, f1_data, labels, colors = [], [], [], []

        for key, exp in self.experiments.items():
            exp_name = exp["name"]
            model_name = exp["model"]

            acc_data.append([f["acc"] for f in exp["folds"]])
            f1_data.append([f["f1"] for f in exp["folds"]])
            labels.append(label_with_features(exp_name, model_name))

            colors.append(MODEL_COLORS.get(model_name, "gray"))

        fig, axes = plt.subplots(1, 2, figsize=(15, 4))  # увеличена ширина
        for ax, data, ylabel, title in zip(
            axes,
            [acc_data, f1_data],
            ["Accuracy", "F1-macro"],
            ["(a) LOSO Accuracy", "(b) LOSO F1-macro"]
        ):
            vp = ax.violinplot(data, showmeans=True, showmedians=True)
            for i, body in enumerate(vp["bodies"]):
                body.set_facecolor(colors[i])
                body.set_alpha(0.7)

            ax.set_xticks(range(1, len(labels) + 1))
            # Увеличено до 45 градусов и горизонтальное выравнивание
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        fig.tight_layout()
        self._save(fig, "fig1_loso_distribution")

    # -------------------------------------------------------------------------
    def fig_subject_variance(self, ref_key=None):
        """
        Fig 2 — inter-subject variability для одной выбранной модели.
        Если ref_key не задан, берём первую модель.
        """
        if ref_key is None:
            ref_key = next(iter(self.experiments.keys()))

        exp = self.experiments[ref_key]
        subjects = [f["subject"] for f in exp["folds"]]
        acc = [f["acc"] for f in exp["folds"]]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(subjects, acc)
        ax.axhline(np.mean(acc), color="red", linestyle="--", label="Mean")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Inter-subject LOSO Variability: {clean_label(exp['model'])}")
        ax.legend()
        ax.set_xticklabels(subjects, rotation=90)

        self._save(fig, f"fig2_subject_variance_{exp['model']}")

    # -------------------------------------------------------------------------
    def fig_subject_accuracy_comparison(self, top_n: int = 3):
        """
        Fig 6 — точность для каждого субъекта (по всем фолдам)
        для top-N наиболее эффективных моделей
        """

        # ---------- Top-N моделей ----------
        model_stats = {}
        for key, exp in self.experiments.items():
            model_name = exp["model"]
            accuracies = [f["acc"] for f in exp["folds"]]
            model_stats[model_name] = {
                "mean_acc": np.mean(accuracies),
                "std_acc": np.std(accuracies),
                "accuracies": accuracies,
                "color": MODEL_COLORS.get(model_name, "gray")
            }

        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: x[1]["mean_acc"],
            reverse=True
        )[:top_n]

        if not sorted_models:
            print("[WARN] No models for comparison")
            return

        # ---------- Figure + GridSpec ----------
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.0, 1.0, 0.45],
            wspace=0.25,
            hspace=0.35,
        )

        ax1 = fig.add_subplot(gs[0, 0:2])   # bar plot
        ax2 = fig.add_subplot(gs[1, 0:2])   # box plot

        ax_leg1 = fig.add_subplot(gs[0, 2])  # legend for bar plot
        ax_leg2 = fig.add_subplot(gs[1, 2])  # legend for stats

        ax_leg1.axis("off")
        ax_leg2.axis("off")

        # ---------- Subjects ----------
        subjects = [f["subject"] for f in self.experiments[next(iter(self.experiments))]["folds"]]
        n_subjects = len(subjects)

        x = np.arange(n_subjects)
        width = 0.25

        model_colors = [m[1]["color"] for m in sorted_models]

        # ---------- (a) Bar plot ----------
        bar_handles = []
        for i, (model_name, stats) in enumerate(sorted_models):

            model_exp_key = next(
                k for k, e in self.experiments.items() if e["model"] == model_name
            )

            acc_by_subject = {
                f["subject"]: f["acc"]
                for f in self.experiments[model_exp_key]["folds"]
            }
            acc_sorted = [acc_by_subject.get(s, 0) for s in subjects]

            offset = width * (i - (top_n - 1) / 2)

            bars = ax1.bar(
                x + offset,
                acc_sorted,
                width,
                color=model_colors[i],
                alpha=0.8,
                label=clean_label(model_name),
            )
            bar_handles.append(bars[0])

        ax1.set_title(f"(a) Per-Subject Accuracy (Top-{top_n})")
        ax1.set_xlabel("Subject")
        ax1.set_ylabel("Accuracy")
        ax1.set_xticks(x)
        ax1.set_xticklabels(subjects, rotation=90)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        # ---------- Bar legend (right, separate cell) ----------
        ax_leg1.legend(
            bar_handles,
            [clean_label(m[0]) for m in sorted_models],
            loc="center",
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True,
            title="Models",
        )

        # ---------- (b) Box plot ----------
        box_data = [model_stats[m[0]]["accuracies"] for m in sorted_models]
        box_labels = [clean_label(m[0]) for m in sorted_models]

        bp = ax2.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black"),
        )

        for patch, color in zip(bp["boxes"], model_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for median in bp["medians"]:
            median.set(color="red", linewidth=2)

        ax2.set_title(f"(b) Accuracy Distribution Across Subjects (Top-{top_n})")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        # ---------- Stats legend (right, separate cell) ----------
        stats_text = [
            f"{clean_label(m[0])}: mean={m[1]['mean_acc']:.3f} ± {m[1]['std_acc']:.3f}"
            for m in sorted_models
        ]

        ax_leg2.legend(
            [plt.Line2D([0], [0], color="black", lw=0) for _ in stats_text],
            stats_text,
            loc="center",
            fontsize=9,
            frameon=True,
            fancybox=True,
            title="Model Statistics",
        )

        self._save(fig, f"fig6_subject_accuracy_comparison_top{top_n}")


    # -------------------------------------------------------------------------
    def fig_learning_curves(self):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1, 1, 0.35],
            wspace=0.25,
            hspace=0.30,
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        ax_leg = fig.add_subplot(gs[:, 2])  # ← занимает обе строки
        ax_leg.axis("off")

        handles = []
        labels = []

        for key, exp in self.experiments.items():
            histories = [f["history"] for f in exp["folds"] if f["history"]]
            if not histories:
                continue

            min_len = min(len(h["train_loss"]) for h in histories)

            tr_loss = np.mean([h["train_loss"][:min_len] for h in histories], axis=0)
            va_loss = np.mean([h["val_loss"][:min_len] for h in histories], axis=0)
            tr_acc  = np.mean([h["train_acc"] [:min_len] for h in histories], axis=0)
            va_acc  = np.mean([h["val_acc"] [:min_len] for h in histories], axis=0)

            epochs = np.arange(1, min_len + 1)
            label = clean_label(exp["model"])
            color = MODEL_COLORS.get(exp["model"], "gray")

            line, = ax1.plot(epochs, tr_loss, color=color, linewidth=1)
            ax2.plot(epochs, va_loss, color=color, linewidth=1)
            ax3.plot(epochs, tr_acc,  color=color, linewidth=1)
            ax4.plot(epochs, va_acc,  color=color, linewidth=1)

            handles.append(line)
            labels.append(label)

        ax1.set_title("(a) Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2.set_title("(b) Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")

        ax3.set_title("(c) Train Accuracy")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, 1)

        ax4.set_title("(d) Validation Accuracy")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy")
        ax4.set_ylim(0, 1)

        if handles:
            ax_leg.legend(
                handles, labels,
                loc="center",
                fontsize=8,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        self._save(fig, "fig3_learning_curves")



    # -------------------------------------------------------------------------
    def fig_confusion_matrix(self, ref_key=None):
        """
        Fig 4 — средняя confusion matrix (counts + normalized recall) для выбранной модели.
        """
        if ref_key is None:
            ref_key = next(iter(self.experiments.keys()))

        exp = self.experiments[ref_key]
        cms = [f["confusion_matrix"] for f in exp["folds"]]
        cm = np.mean(cms, axis=0)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        labels = exp["folds"][0]["class_names"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axes[0].imshow(cm, cmap="Blues")
        im2 = axes[1].imshow(cm_norm, cmap="RdYlGn", vmin=0, vmax=1)

        for ax, title in zip(axes, ["Counts", "Recall (norm)"]):
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_title(title)

        fig.colorbar(im1, ax=axes[0])
        fig.colorbar(im2, ax=axes[1])
        self._save(fig, f"fig4_confusion_matrix_{exp['model']}")

    # -------------------------------------------------------------------------
    def fig_acc_f1_scatter(self):
        """
        Fig 5 — scatter Accuracy vs F1-макро по fold'ам для всех моделей.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        for key, exp in self.experiments.items():
            color = MODEL_COLORS.get(exp["model"], None)
            ax.scatter(
                [f["acc"] for f in exp["folds"]],
                [f["f1"] for f in exp["folds"]],
                label=clean_label(exp["model"]),
                alpha=0.7,
                color=color
            )
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("F1-macro")
        ax.legend(fontsize=7)
        ax.set_title("Accuracy vs F1 (LOSO)")
        self._save(fig, "fig5_acc_f1_scatter")

    # -------------------------------------------------------------------------
    def _save(self, fig, name):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{name}.pdf")
        fig.savefig(self.output_dir / f"{name}.png")
        print("[FIG]", self.output_dir / f"{name}.png")
        plt.close(fig)

    # -------------------------------------------------------------------------
    def generate_all(self, ref_key_for_subject=None, ref_key_for_cm=None, top_n_comparison: int = 3):
        self.fig_loso_distribution()
        self.fig_subject_variance(ref_key=ref_key_for_subject)
        self.fig_learning_curves()
        self.fig_confusion_matrix(ref_key=ref_key_for_cm)
        self.fig_acc_f1_scatter()
        self.fig_subject_accuracy_comparison(top_n=top_n_comparison)


# =============================================================================
# Таблицы для статьи
# =============================================================================
def summarize_models(df_folds: pd.DataFrame, out_dir: Path):
    """
    Создаёт:
    - Table 1: все модели
    - Table 2: deep-модели (exp1, exp3, exp5)
    - Table 3: ML-модели (exp4 или is_ml)
    В формате CSV и LaTeX.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_folds.empty:
        print("[WARN] df_folds is empty, no tables created.")
        return

    agg = (
        df_folds
        .groupby(["exp_name", "model_name"])
        .agg(
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("f1_macro", "mean"),
            std_f1=("f1_macro", "std"),
            n_subjects=("subject", "nunique"),
            is_ml=("is_ml", "max"),
        )
        .reset_index()
    )

    def detect_features(row):
        exp = row["exp_name"]
        model = row["model_name"]
        # По умолчанию: Raw EMG
        features = "Raw EMG"
        if "emg_td" in exp or "powerful" in exp:
            features = "TD features"
        if "svm" in model or "rf" in model:
            features = "TD features"
        return features

    agg["features"] = agg.apply(detect_features, axis=1)
    agg["model_short"] = agg["model_name"].apply(clean_label)
    agg["model_full"] = agg["model_name"].apply(clean_label)

    # Таблица 1 — все модели
    table1 = agg[[
        "model_short", "model_full", "features",
        "mean_acc", "std_acc", "mean_f1", "std_f1", "n_subjects"
    ]].sort_values(["features", "mean_acc"], ascending=[True, False])

    table1.to_csv(out_dir / "table1_all_models_loso.csv", index=False)
    with open(out_dir / "table1_all_models_loso.tex", "w") as f:
        f.write(table1.to_latex(index=False, float_format="%.3f"))

    # Таблица 2 — deep-модели (exp1, exp3, exp5 и не ML)
    deep_mask = (~agg["is_ml"]) & agg["exp_name"].str.startswith(("exp1_", "exp3_", "exp5_"))
    table2 = agg.loc[deep_mask, [
        "model_short", "model_full", "features",
        "mean_acc", "std_acc", "mean_f1", "std_f1", "n_subjects"
    ]].sort_values("mean_acc", ascending=False)
    table2.to_csv(out_dir / "table2_deep_models_loso.csv", index=False)
    with open(out_dir / "table2_deep_models_loso.tex", "w") as f:
        f.write(table2.to_latex(index=False, float_format="%.3f"))

    # Таблица 3 — ML-модели (exp4 или is_ml)
    ml_mask = agg["is_ml"] | agg["exp_name"].str.startswith(("exp4_",))
    table3 = agg.loc[ml_mask, [
        "model_short", "model_full", "features",
        "mean_acc", "std_acc", "mean_f1", "std_f1", "n_subjects"
    ]].sort_values("mean_acc", ascending=False)
    table3.to_csv(out_dir / "table3_ml_models_loso.csv", index=False)
    with open(out_dir / "table3_ml_models_loso.tex", "w") as f:
        f.write(table3.to_latex(index=False, float_format="%.3f"))

    print("[TABLES] Saved to", out_dir)


def classwise_metrics_for_experiment(df_folds: pd.DataFrame,
                                     exp_name: str,
                                     model_name: str,
                                     out_dir: Path):
    """
    Строит class-wise таблицу (precision, recall, f1) по усреднённой confusion matrix
    для заданного (exp_name, model_name).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = (df_folds["exp_name"] == exp_name) & (df_folds["model_name"] == model_name)
    df_exp = df_folds.loc[mask]

    if df_exp.empty:
        print(f"[WARN] No data for exp={exp_name}, model={model_name}")
        return

    # Предполагаем одинаковый порядок классов
    class_names = df_exp.iloc[0]["class_names"]
    cms = np.stack(df_exp["confusion_matrix"].values, axis=0)
    cm_sum = cms.sum(axis=0)  # shape (C, C)

    tp = np.diag(cm_sum)
    support = cm_sum.sum(axis=1)         # по строкам — реальные классы
    pred_support = cm_sum.sum(axis=0)    # по столбцам — предсказанные

    recall = tp / np.maximum(support, 1)
    precision = tp / np.maximum(pred_support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

    df_cls = pd.DataFrame({
        "class": class_names,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    df_cls.to_csv(out_dir / f"classwise_{exp_name}_{model_name}.csv", index=False)
    with open(out_dir / f"classwise_{exp_name}_{model_name}.tex", "w") as f:
        f.write(df_cls.to_latex(index=False, float_format="%.3f"))

    print("[CLASSWISE]", exp_name, model_name, "->", out_dir)


# =============================================================================
# Построение структуры experiments для визуализатора
# =============================================================================
def build_experiments_structure(df_folds: pd.DataFrame) -> dict:
    """
    Преобразует df_folds в формат, ожидаемый PaperVisualizationGenerator.
    Ключ словаря — (exp_name, model_name).
    """
    experiments = {}
    for (exp_name, model_name), df_grp in df_folds.groupby(["exp_name", "model_name"]):
        folds = []
        for _, row in df_grp.iterrows():
            folds.append({
                "subject": row["subject"],
                "acc": row["accuracy"],
                "f1": row["f1_macro"],
                "confusion_matrix": row["confusion_matrix"],
                "class_names": row["class_names"],
                "history": row.get("history", None),
            })
        key = (exp_name, model_name)
        experiments[key] = {
            "name": exp_name,
            "model": model_name,
            "summary": None,
            "folds": folds,
        }
    return experiments

def label_with_features(exp_name: str, model_name: str) -> str:
    base = clean_label(model_name)
    # простая логика, можно доработать
    if "emg_td" in exp_name or "powerful" in exp_name or "svm" in model_name or "rf" in model_name:
        feats = "TD"
    else:
        feats = "Raw"
    return f"{base} ({feats})"

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Путь к артефактам и куда сохранять фигуры/таблицы
    base_dir = Path("/home/kirill/projects_2/folium/NIR/OMEGA/experiments_output")
    paper_root = Path("/home/kirill/projects_2/folium/NIR/OMEGA")
    paper_figures = paper_root / "paper_figures"
    tables_dir = paper_figures / "tables"
    classwise_dir = paper_figures / "classwise"

    # Список экспериментов (как у тебя в исходном коде)
    experiments = [
        'exp1_deep_raw_attention_cnn_loso_isolated_v2',
        'exp1_deep_raw_bilstm_attention_loso_isolated_v2',
        'exp1_deep_raw_bilstm_loso_isolated_v2',
        'exp1_deep_raw_multiscale_cnn_loso_isolated_v2',
        'exp1_deep_raw_simple_cnn_loso_isolated_v2',
        'exp1_deep_raw_tcn_attn_loso_isolated_v2',
        'exp1_deep_raw_tcn_loso_isolated_v2',
        'exp1_deep_raw_bigru_loso_isolated_v2',
        'exp1_deep_raw_cnn_gru_attention_loso_isolated_v2',
        'exp1_deep_raw_cnn_lstm_loso_isolated_v2',
        'exp2_deep_emg_td_seq_attention_cnn_loso',
        'exp2_deep_emg_td_seq_bigru_loso',
        'exp2_deep_emg_td_seq_bilstm_attention_loso',
        'exp2_deep_emg_td_seq_bilstm_loso',
        'exp2_deep_emg_td_seq_cnn_gru_attention_loso',
        'exp2_deep_emg_td_seq_cnn_lstm_loso',
        'exp2_deep_emg_td_seq_multiscale_cnn_loso',
        'exp2_deep_emg_td_seq_simple_cnn_loso',
        'exp2_deep_emg_td_seq_tcn_attn_loso',
        'exp2_deep_emg_td_seq_tcn_loso',
        'exp3_deep_powerful_mlp_powerful_loso',
        'exp4_svm_linear_powerful_loso',
        'exp4_svm_rbf_powerful_loso',
        'exp4_rf_powerful_loso',
        'exp5_hybrid_powerful_deep_loso',
        'exp6_sota_best_models_aug_loso'
    ]

    # 1) Агрегация
    agg = ExperimentAggregator(base_dir)
    agg.load_all(experiments)
    df_folds = agg.to_fold_df()

    # 2) Таблицы
    summarize_models(df_folds, tables_dir)

    # Пример: class-wise таблица для SVM-RBF (exp4_svm_rbf_powerful_loso / svm_rbf)
    classwise_metrics_for_experiment(
        df_folds,
        exp_name="exp4_svm_rbf_powerful_loso",
        model_name="svm_rbf",
        out_dir=classwise_dir
    )

    # Можно добавить ещё несколько class-wise таблиц для лучших deep/ML моделей по аналогии

    # 3) Фигуры
    exp_struct = build_experiments_structure(df_folds)

    # Выберем какую-нибудь референтную модель для:
    # - Fig 2 (inter-subject variability)
    # - Fig 4 (confusion matrix)
    # Например, SVM-RBF:
    ref_key = None
    for k in exp_struct.keys():
        if "exp4_svm_rbf_powerful_loso" in k[0]:
            ref_key = k
            break
    if ref_key is None:
        # если SVM-RBF нет, берём первую попавшуюся
        ref_key = next(iter(exp_struct.keys()))

    gen = PaperVisualizationGenerator(paper_figures, exp_struct)
    gen.generate_all(
        ref_key_for_subject=ref_key, 
        ref_key_for_cm=ref_key,
        top_n_comparison=3  # или любое другое число
    )
    print("Done.")