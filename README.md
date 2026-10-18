# Azerbaijani Property Price Prediction

MLE-AI-201 Machine Learning final project using bina.az property-sale listings.

The project implements a decision tree and a Pegasos SVM from scratch in NumPy and compares them with scikit-learn baselines. Additional implementations include random-forest regression, Ridge regression, and Pegasos-style SVR.

## Tasks

- **Regression:** predict the advertised sale price.
- **Classification:** predict standard (`0`) or premium (`1`), with premium defined as a price above the training-set median.

Regression experiments use both raw prices and `log1p(price)` targets. Log-target predictions are converted back to AZN before RMSE, MAE, and R² are calculated.

## Models

| Model | Implementation |
|-------|----------------|
| Decision tree | Gini, entropy, MSE, recursive threshold splits, stopping rules, probabilities, feature importances |
| Pegasos SVM | Regularised hinge loss, stochastic updates, training-only scaling |
| Nonlinear SVM | Random Fourier Features approximating an RBF kernel |
| Random forest | Bootstrap samples, per-node feature subsampling, averaged regression predictions |
| Ridge regression | From-scratch regularised linear regression |
| Pegasos-style SVR | Sub-gradient training with epsilon-insensitive loss |

Scikit-learn model classes are used for comparison baselines, not inside the custom model implementations.

## Setup

Install Git and Git LFS first. Use **Python 3.11** with the pinned dependencies. Python 3.13 is incompatible with the pinned NumPy 1.26.4 installation.

### Windows PowerShell

```powershell
git clone https://github.com/nazrinburz/mle-ai-final-project.git
cd mle-ai-final-project
git lfs install --local
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.11` cannot find an installed interpreter, install/register Python 3.11 or use its explicit executable path to create the virtual environment.

### Linux / macOS

With Python 3.11 installed:

```bash
git clone https://github.com/nazrinburz/mle-ai-final-project.git
cd mle-ai-final-project
git lfs install --local
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements.txt
```

## Dataset

Source: [Binaaz Sale Project by sehriyarmemmedli](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project).

The default evaluation input is:

```text
data/house_sale.csv
```

The exact joined export is recoverable from historical Git LFS storage; it is not tracked in the current revision. After installing Git LFS and completing the environment/dependency setup above, run from the repository root:

```powershell
.\.venv311\Scripts\python.exe -m src.fetch_dataset
```

This fetches `data/house_sale.csv` from commit `419ebacf59de3723fa7252fed7d972badcffe583`, verifies SHA-256, and writes `data/house_sale.csv` without adding it to Git. No join, conversion, or resaving is needed: it retrieves the exact prepared export. The helper refuses to overwrite a file with a different hash. Git LFS server access must remain available.

Expected SHA-256: `c9ec2ded82c2e93ab7ad13ad6aed2f6d540d890c3d8d089a8ce9fdb65a3bb7e6` (141,782,218 bytes). Retrieval was tested on 18 October 2026 into isolated temporary storage with `python -m src.fetch_dataset --output tmp/retrieved/house_sale.csv`; the hash and byte count matched. The helper fetches the historical commit automatically if a shallow clone lacks it. A fresh shallow-clone test also passed after local Git LFS setup and pinned dependency installation; no existing LFS cache was copied. See [data/README.md](data/README.md) for access and fallback requirements.

The Kaggle link identifies the original source, not a verified replacement CSV. No deterministic original join recipe was found in repository scripts/history. If historical LFS becomes unavailable, obtain the exact export from the data owner, or obtain the source archive version and filenames, join keys/type/cardinality, duplicate ordering, transformations, and export/encoding rules needed to reconstruct it. Do not infer these from column suffixes. Use `--data path/to/house_sale.csv` for another local location.

Keep Azerbaijani column names correctly encoded in UTF-8. Do not commit downloaded dataset archives or redistribute personal listing information.

## Run

Run commands from the repository root.

### Full evaluation — Windows

```powershell
.\.venv311\Scripts\python.exe -m src.run_all --output report/reproduced-evaluation --figures report/reproduced-figures
```

### Full evaluation — Linux / macOS

```bash
.venv311/bin/python -m src.run_all --output report/reproduced-evaluation --figures report/reproduced-figures
```

The full run performs validation-based model selection, matched baseline comparisons, held-out evaluation, and training-only cross-validation. Exhaustive custom-tree split searches can make the full run slow.

These commands use the full dataset, 20,000 iterations, seed 42, and three-fold training-only cross-validation by default. Obtain the input CSV first as described above. Use `--data path/to/house_sale.csv` if it is stored elsewhere. Do not launch another full evaluation while one is already running. The separate reproduction directories preserve saved final results; choose unused output paths if these directories already contain a run to retain.

### Reproduction audit and remaining steps

The saved final results are in `report/evaluation-clean-log/`, including `metrics.csv`, `manifest.json`, and saved prediction/error summaries. The saved final headline metrics match the report at its displayed precision: raw tree RMSE 71,741 AZN, MAE 48,559 AZN and R² 0.660; log tree RMSE 74,011 AZN; tree classifier F1 0.873 and ROC-AUC 0.943. The final manifest records 59,697 cleaned rows, a 41,787 / 8,955 / 8,955 split, 167 features, a 195,000 AZN premium threshold, and shared `data_prep.clean()` before splitting.

`report/evaluation/` contains older results from a different cleaning protocol. Its metrics must not be used to assess agreement with the final report or combined with the final run's outputs.

**Saved final metrics match the report; a fresh full end-to-end reproduction has not been verified.** No new full evaluation was run for this audit. Once the matching dataset is available and any active evaluation has finished, run the command above and compare the new manifest, metrics (including raw/log models), selected settings, CV outputs, and generated figures with the saved final evidence. Runtime measurements vary with hardware and load.

The command generates the report's four figure filenames: `regression_comparison.png`, `classification_comparison.png`, `feature_importance_Tree_regression.png`, and `roc.png`. Existing copies are in `report/figures/`; the reproduction command writes new copies to `report/reproduced-figures/`. File existence alone does not verify a fresh numerical reproduction.

The command now emits `high_price_error_summary.csv`: each regression model's count and mean absolute error for `true>500000`. Verified final tree errors give 307 properties and MAE 169,358.57 AZN. The report's roughly 165,000 AZN statement is inconsistent with those errors and cannot be reproduced exactly; the report has not been edited. See [the claim-to-code map](report/reproduction.md) for all headline tables, narrative numbers, and figures.

PDF compilation is separate from scientific result reproduction. The evaluation command does not compile the report or slides, and that alone is not evidence that model results are irreproducible. To rebuild the report document, compile `report/report.tex` with a LaTeX toolchain (for example, run `pdflatex report.tex` twice from `report/`). Editable slide source and its build instructions are not present in `presentation/` and would be needed to regenerate the presentation PDF.

### Small pipeline check

Use separate output locations to preserve full-run results:

```powershell
.\.venv311\Scripts\python.exe -m src.run_all --max-rows 400 --iterations 1000 --skip-cv --output report/pilot_evaluation --figures report/pilot_figures
```

This uses sampled real data and is a smoke check, not a substitute for full-data results.

### Other options

```powershell
.\.venv311\Scripts\python.exe -m src.run_all --help
```

Supported options include `--data`, `--output`, `--figures`, `--max-rows`, `--iterations`, `--skip-cv`, and `--cv-folds`.

## Evaluation

- **Regression:** RMSE, MAE, and R².
- **Classification:** precision, recall, F1, ROC-AUC, and confusion matrices.
- **Model selection:** validation RMSE for regression and validation F1 for classification.
- **Analysis:** prediction errors, feature importances, SVM weights, objective histories, and runtime.
- **Uncertainty:** paired test-bootstrap intervals and a separate training-only cross-validation summary.

Bootstrap intervals and cross-validation standard deviations represent different sources of uncertainty and should not be interpreted interchangeably.

Saved results and output locations:

| Location | Contents |
|----------|----------|
| `report/evaluation-clean-log/` | Saved final metrics, manifest, predictions, error summaries, and other evaluation outputs supporting the report |
| `report/figures/` | Existing report figures |
| `report/reproduced-evaluation/` | New metrics, validation searches, predictions, error summaries, importance tables, manifest, and CV outputs from the reproduction command |
| `report/reproduced-figures/` | New comparison plots, confusion matrices, ROC curves, feature importances, and objective curves |
| `report/evaluation/` | Older evaluation outputs; also the script default when `--output` is omitted |

## Data Protocol and Limitations

The pipeline removes repeated listing snapshots before modelling and uses a deterministic 70/15/15 split with seed 42. The validation/test subdivision is stratified using labels based on the training-price median.

The current pipeline calls the shared `data_prep.clean()` function before splitting. Any data-dependent cleaning decisions made at that stage are a limitation of the evaluation protocol. Feature encoding and imputation are fitted on training data, and custom SVM scaling uses training statistics only.

Target-derived columns such as `unit_price` and `total_price` are excluded from model inputs. Hyperparameters are selected using validation data.

The targets are advertised listing prices, not confirmed transaction prices. Reported results are conditional on the dataset, cleaning rules, and evaluation protocol recorded for that run.

## Tests

```powershell
.\.venv311\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Tests include tiny decision-tree comparisons against scikit-learn and checks for the custom forest and other model components. Passing tests establish the checked behaviours; they do not establish real-data predictive performance.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── contribution_report.md
├── data/
├── src/
│   ├── data_prep.py
│   ├── decision_tree.py
│   ├── svm.py
│   ├── svm_experiments.py
│   ├── linear_baseline.py
│   ├── random_forest.py
│   ├── evaluate.py
│   └── run_all.py
├── tests/
├── report/
│   ├── report.pdf
│   ├── evaluation/
│   └── figures/
└── presentation/
    └── slides.pdf
```

## Team

| Member | Main area |
|--------|-----------|
| Gunel Garayzade | Evaluation, metrics, experiments, result analysis |
| Jale Afandiyeva | SVM and linear baselines |
| Kheyransa Gorchuyeva | Decision tree, random forest, testing |
| Nazrin Burziyeva | Data preparation, integration, documentation, report and presentation |

See [contribution_report.md](contribution_report.md) for the detailed contribution breakdown.

## Report and Presentation

- Report: [report/report.pdf](report/report.pdf)
- Presentation: [presentation/slides.pdf](presentation/slides.pdf)

External resources and AI assistance are acknowledged in the report.
