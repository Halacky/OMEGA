# Codegen Rules for Experiment Generation

These rules are MANDATORY for all generated experiment code. Violating any of them
will cause runtime errors, data loading failures, or incorrect results.

---

## 1. Subject List Handling (CRITICAL)

- Subject list MUST be overridable via `parse_subjects_args()` from `exp_X_template_loso`.
- **Default MUST be CI subjects** (`DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39`), NOT a full 20-subject list.
- The vast.ai server has symlinks ONLY for CI subjects. Using full subject list on server causes `FileNotFoundError`.
- Pattern: `ALL_SUBJECTS = parse_subjects_args()` in `main()`.
- NEVER hardcode arbitrary 20-subject lists as defaults.

## 2. Data Loading API

- `load_multiple_subjects(base_dir=, subject_ids=, exercises=, include_rest=)` exact kwargs.
  - kwarg is `subject_ids=`, NOT `subjects=`.
  - Returns `Dict[str, Tuple[emg, segments, grouped_windows]]` dict of tuples, NOT dict of dicts.
- `load_subject(base_dir=, subject_id=, exercise=)` exercise is a single string (e.g. `"E1"`), NOT a list.
- Returns **tuple**: `(emg, segments, grouped_windows)` NOT a dict.
- Unpack as: `for subj_id, (emg, segments, grouped_windows) in data.items()`.
- Use `grouped_to_arrays(grouped_windows)` to get `(windows, labels)` from grouped_windows.
- NEVER access result as `subj_data["windows"]` or `subj_data.get("windows")` it is a tuple.
- Method `load_subjects(cs_cfg)` does NOT exist. Method `_load_subjects_data()` does NOT exist.

## 3. Feature Extraction

- Import: `from processing.powerful_features import PowerfulFeatureExtractor`
- NOT `from features.*` (module doesn't exist, LLM hallucinates it).
- NEVER redefine `PowerfulFeatureExtractor` locally always import.
- Constructor: `PowerfulFeatureExtractor(sampling_rate=2000)`.
- `.transform(X)` expects shape `(N, T, C)` windows from `grouped_windows` are already `(N, T, C)`.
- NO transpose needed when using `grouped_windows` directly.
- Always `handcrafted_dim = features.shape[1]` after extraction, NEVER hardcode feature dimensions.

## 4. Data Shape

- Windows stored in `grouped_windows` are shape `(N, T, C)` NOT `(N, C, T)`.
- Splits in `CrossSubjectExperiment` are also `(N, T, C)`.
- `PowerfulFeatureExtractor.transform()` takes `(N, T, C)` NO transpose needed.

## 5. PyTorch Compatibility

- `ReduceLROnPlateau(..., verbose=True)` `verbose` was removed in PyTorch 2.4+. NEVER use it.

## 6. Format String Safety

- ALWAYS guard `.4f` formatting: `if val is not None` before `f"{val:.4f}"`.
- Values from evaluation can be `None` formatting `None` as `.4f` crashes.

## 7. Argparse in main()

- In `main()`, use `parse_known_args()` (not `parse_args()`) so runner scripts can pass extra flags.
- Use `_parser` / `_args` / `_CI_SUBJECTS` / `_FULL_SUBJECTS` names to avoid polluting namespace.

## 8. Error Handling

- NEVER use bare `except Exception as e: print(e)` in data loading loops.
- Always re-raise: `logger.error(...); raise` otherwise exceptions are swallowed silently.
- Swallowed errors lead to misleading secondary errors like "need at least one array to concatenate".

## 9. Class Index Mapping

- After `load_multiple_subjects`, `gesture_ids` are NOT class indices (0,1,2...).
- Must build: `gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}`.
- Use `multi_loader.get_common_gestures(subjects_data, max_gestures=N)` for common gestures.

## 10. Cleanup Code

- Verify `del` cleanup references correct variable names.
- Do NOT `del subjects_data` if data is stored in `subjects_windows` / `subjects_labels`.

## 11. Custom Trainer Interface

- Custom trainers passed to `CrossSubjectExperiment` MUST implement:
  - `fit(splits)` not just `train(train_windows, ...)`.
  - `evaluate_numpy(X, y, split_name, visualize)`.
  - `self.class_ids` attribute.
- If custom trainer has `train()`, add a `fit()` adapter that unpacks splits, runs feature extraction, sets `self.class_ids`.

## 12. Model Registration

- For new models, use `from models import register_model` and call `register_model("name", Class)`.
- Do NOT modify `models/__init__.py` directly.

## 13. File Paths

- `BASE_DIR = ROOT / "data"` NEVER hardcoded absolute paths.
- `OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")`.

## 14. Qdrant Callbacks

- Define `HYPOTHESIS_ID = "<value>"` right after `EXPERIMENT_NAME`.
- Import `mark_hypothesis_verified`, `mark_hypothesis_failed` from `hypothesis_executor.qdrant_callback`.
- Call at END of `main()` after saving `loso_summary.json`.

## 15. Template Compliance

- Always import and use `from experiments.exp_X_template_loso import run_single_loso_fold, make_json_serializable`.
- Only rewrite `run_single_loso_fold` locally if the hypothesis requires custom training loop modifications.
- Include `if __name__ == "__main__": main()` guard.
- `EXERCISES = ["E1"]`.
- `pipeline_type` must be one of: `deep_raw`, `deep_emg_td_seq`, `ml_emg_td`.
