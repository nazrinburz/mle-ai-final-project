# Scientific reproduction map

Run from the repository root after retrieving the exact dataset and installing pinned requirements:

```powershell
.\.venv311\Scripts\python.exe -m src.run_all --output report/reproduced-evaluation --figures report/reproduced-figures
```

On Linux/macOS substitute `.venv311/bin/python`. The defaults retain the final run's seed 42, full data, 20,000 iterations, validation candidate grids and three training-only CV folds. The model/data/pipeline code at the final-report commit matches the starting code here; no model/configuration change was needed. New output paths protect `report/evaluation-clean-log/` and `report/figures/`. Do not reuse previous files in new output directories or run concurrently with an existing expensive evaluation.

| Report claims | Generating code in `src/run_all.py` | New output |
|---|---|---|
| 100,775 source rows; 36,321 repeated rows; 59,697 cleaned rows; 41,787/8,955/8,955 split; 167 features; median threshold 195,000 | `main`: load, listing deduplication, shared `clean`, seed-42 split, `prepare_splits` | `manifest.json`: counts, feature list, threshold, dataset hash |
| Near-balanced tiers | `prepare_splits`: compare targets with training median | Manifest `tier_counts` for train/validation/test, ordered standard/premium |
| Selected depth/leaf/criterion, including raw depth 6/leaf 15 and entropy classification | `main`: validation candidate search; fixed seeds | `validation_search.csv`, manifest `selected_parameters` |
| Table 2: every custom/sklearn regression row; raw/log comparison and narrative R2/MAE; custom SVR 609,574 / -23.518 | `main`: selected estimators, inverse-transform log predictions before `regression_metrics` | `metrics.csv`, `comparison_table.csv`, per-model `errors_*.csv` |
| Table 3: all custom/sklearn precision, recall, F1, ROC-AUC | `main`: class predictions and scores, `classification_metrics` | `metrics.csv`, predictions CSVs, `classification_error_summary.csv` |
| Table 4: all fit/predict times | `main`: `perf_counter` around selected fit and prediction | `metrics.csv`; hardware/load-dependent, exact times not reproducible constants |
| Tree median absolute error 32,651 AZN | `analyze_saved_results`: median saved absolute error | `regression_error_summary.csv` |
| Price above 500,000 AZN error | `analyze_saved_results`: strictly filter true price >500000 | `high_price_error_summary.csv`, including count and MAE for every regression model |
| Figure 1 raw/log comparison | `save_summary_plots` | `regression_comparison.png` |
| Figure 2 classification comparison | `save_summary_plots` | `classification_comparison.png` |
| Figure 3 selected tree importance | `main` and `analyze_saved_results` | `feature_importance_Tree_regression.png`, importance CSV |
| Figure 4 ROC | `main`: scores and `roc_curve` | `roc.png` |
| Three-fold training-only check; paired bootstrap intervals | `cross_validate_training` (called by default); `main` bootstrap loop | `cv_folds.csv`, `cv_summary.csv`, `cv_protocol.json`, `metric_intervals.csv` |

Saved final results match the report's main metric tables at displayed precision. A read-only calculation from final tree errors gives **307** properties above 500,000 AZN and **169,358.56983111717 AZN** MAE. The narrative's roughly 165,000 AZN is not the value generated from those errors. It needs a report correction; this work leaves report source/PDF unchanged. Claims such as forest stability are qualitative and are not established by a dedicated repeated-seed experiment here.

The dataset retrieval/hash was verified and a small fresh full-pipeline smoke check is separate from full-data verification. A fresh full end-to-end run has not been verified; do not present saved final results or smoke results as a fresh reproduction. Original source-to-joined-export preparation remains undocumented if the historical exact object is lost.

PDF compilation is a separate document workflow. `run_all` generates numerical results and image assets, not report prose, LaTeX tables or slides. Rebuild the report with a LaTeX toolchain only after reviewing verified outputs. Missing PDF build commands alone do not establish scientific irreproducibility. Editable slide source is not present.

Fresh-clone retrieval validation (18 October 2026): cloned origin with `--depth 1`, copied only the candidate fetch helper into that checkout, ran `git lfs install --local`, created a fresh Python 3.11 virtual environment and installed the unchanged pinned requirements. Git Credential Manager was configured; public LFS access needed no interactive prompt. Running `.\.venv311\Scripts\python.exe -m src.fetch_dataset` automatically fetched the missing historical commit and downloaded a new LFS object into that clone. The resulting default `data/house_sale.csv` matched the expected 141,782,218 bytes and SHA-256. No existing LFS object store was copied. This tests the candidate helper with a fresh clone, not a full training reproduction. On the test host `py -3.11` was unregistered, so an existing Python 3.11 interpreter created the fresh environment; users need a registered Python 3.11 installation or its explicit executable path.
