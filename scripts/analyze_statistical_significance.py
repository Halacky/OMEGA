# create: scripts/analyze_statistical_significance.py

import json
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

# Модели для сравнения
models = {
    'BiGRU': 'exp1_deep_raw_bigru_loso_isolated_v2/bigru',
    'Simple_CNN': 'exp1_deep_raw_simple_cnn_loso_isolated_v2/simple_cnn',
    'Multiscale_CNN': 'exp1_deep_raw_multiscale_cnn_loso_isolated_v2/multiscale_cnn',
    'CNN_GRU_Attn': 'exp1_deep_raw_cnn_gru_attention_loso_isolated_v2/cnn_gru_attention',
    'Attention_CNN': 'exp1_deep_raw_attention_cnn_loso_isolated_v2/attention_cnn',
    'BiLSTM_Attn': 'exp1_deep_raw_bilstm_attention_loso_isolated_v2/bilstm_attention',
}

ALL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]

# Собрать accuracy для каждой модели
model_accuracies = {}

for model_name, model_path in models.items():
    accuracies = []
    for subject in ALL_SUBJECTS:
        result_file = Path(f"experiments_output/{model_path}/test_{subject}/cross_subject_results.json")
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                acc = data['cross_subject_test']['accuracy']
                accuracies.append(acc)
    
    model_accuracies[model_name] = np.array(accuracies)
    print(f"{model_name}: {len(accuracies)} subjects, mean={np.mean(accuracies):.4f}")

# Pairwise t-tests
print("\n=== Pairwise t-tests (paired) ===")
for (model1, acc1), (model2, acc2) in combinations(model_accuracies.items(), 2):
    t_stat, p_value = stats.ttest_rel(acc1, acc2)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    print(f"{model1:15s} vs {model2:15s}: t={t_stat:6.3f}, p={p_value:.4f} {sig}")