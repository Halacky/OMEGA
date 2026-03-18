import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

model_path = "exp1_deep_raw_bigru_loso_isolated_v2/bigru"
ALL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]

# Агрегировать CM
cm_total = None
for subject in ALL_SUBJECTS:
    result_file = Path(f"experiments_output/{model_path}/test_{subject}/cross_subject_results.json")
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            cm = np.array(data['cross_subject_test']['confusion_matrix'])
            if cm_total is None:
                cm_total = cm
            else:
                cm_total += cm

# Анализ ошибок
total_samples = cm_total.sum()
correct = np.trace(cm_total)
total_errors = total_samples - correct

print(f"Total samples: {total_samples}")
print(f"Correct: {correct} ({correct/total_samples*100:.1f}%)")
print(f"Total errors: {total_errors}")

# Группы жестов (индексы 0-9 соответствуют жестам 8-17)
rotation_indices = [1, 2, 3, 4]  # жесты 9,10,11,12
flexion_indices = [5, 6, 9]      # жесты 13,14,17
deviation_indices = [7, 8]        # жесты 15,16
finger_indices = [0]              # жест 8

# Ошибки внутри групп
def errors_within_group(cm, indices):
    errors = 0
    for i in indices:
        for j in indices:
            if i != j:
                errors += cm[i, j]
    return errors

rotation_errors = errors_within_group(cm_total, rotation_indices)
flexion_errors = errors_within_group(cm_total, flexion_indices)
deviation_errors = errors_within_group(cm_total, deviation_indices)

print(f"\n=== Within-Group Errors ===")
print(f"Rotation group (9-12): {rotation_errors} ({rotation_errors/total_errors*100:.1f}%)")
print(f"Flexion group (13-14-17): {flexion_errors} ({flexion_errors/total_errors*100:.1f}%)")
print(f"Deviation group (15-16): {deviation_errors} ({deviation_errors/total_errors*100:.1f}%)")

# Ошибки между группами
def errors_between_groups(cm, group1, group2):
    errors = 0
    for i in group1:
        for j in group2:
            errors += cm[i, j] + cm[j, i]
    return errors

rotation_flexion_errors = errors_between_groups(cm_total, rotation_indices, flexion_indices)
rotation_deviation_errors = errors_between_groups(cm_total, rotation_indices, deviation_indices)

print(f"\n=== Between-Group Errors ===")
print(f"Rotation ↔ Flexion: {rotation_flexion_errors} ({rotation_flexion_errors/total_errors*100:.1f}%)")
print(f"Rotation ↔ Deviation: {rotation_deviation_errors} ({rotation_deviation_errors/total_errors*100:.1f}%)")

# Топ-5 самых частых ошибок
cm_errors = cm_total.copy()
np.fill_diagonal(cm_errors, 0)  # убрать диагональ
top_errors = []
for i in range(10):
    for j in range(10):
        if i != j:
            top_errors.append((i+8, j+8, cm_errors[i, j]))  # +8 чтобы получить номера жестов
top_errors.sort(key=lambda x: x[2], reverse=True)

print(f"\n=== Top-5 Most Frequent Confusions ===")
for true_g, pred_g, count in top_errors[:5]:
    print(f"Gesture {true_g} → {pred_g}: {count} ({count/total_errors*100:.1f}%)")

# Визуализация
cm_norm = cm_total.astype('float') / cm_total.sum(axis=1, keepdims=True)

plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=[f'G{i}' for i in range(8, 18)],
            yticklabels=[f'G{i}' for i in range(8, 18)],
            cbar_kws={'label': 'Normalized Frequency'})
plt.title('Aggregated Confusion Matrix: BiGRU on Raw EMG (20 LOSO folds)', fontsize=14)
plt.ylabel('True Gesture', fontsize=12)
plt.xlabel('Predicted Gesture', fontsize=12)

# Добавить линии для группировки
for pos in [1.5, 5.5, 7.5, 9.5]:
    plt.axhline(y=pos, color='red', linewidth=2, linestyle='--', alpha=0.5)
    plt.axvline(x=pos, color='red', linewidth=2, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('confusion_matrix_bigru_grouped.png', dpi=300, bbox_inches='tight')
plt.show()