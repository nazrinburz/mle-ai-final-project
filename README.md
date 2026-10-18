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

Use **Python 3.11** with the pinned dependencies. Python 3.13 is incompatible with the pinned NumPy 1.26.4 installation.

### Windows PowerShell

```powershell
git clone https://github.com/nazrinburz/mle-ai-final-project.git
cd mle-ai-final-project
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux / macOS

With Python 3.11 installed:

```bash
git clone https://github.com/nazrinburz/mle-ai-final-project.git
cd mle-ai-final-project
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements.txt
```

## Dataset

Source: [Binaaz Sale Project by sehriyarmemmedli](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project).

The default evaluation input is:

```text
data/house_sale.csv
```

Ensure this is the actual CSV rather than a Git LFS pointer. If the repository provides it through Git LFS, install Git LFS and run:

```bash
git lfs pull
```

Alternatively, obtain the dataset from the source and supply the corresponding input CSV using `--data`. The input must match the column schema expected by `src/data_prep.py`.

Keep Azerbaijani column names correctly encoded in UTF-8. Do not commit downloaded dataset archives or redistribute personal listing information.

## Run

Run commands from the repository root.

### Full evaluation — Windows

```powershell
.\.venv311\Scripts\python.exe -m src.run_all
```

### Full evaluation — Linux / macOS

```bash
.venv311/bin/python -m src.run_all
```

The full run performs validation-based model selection, matched baseline comparisons, held-out evaluation, and training-only cross-validation. Exhaustive custom-tree split searches can make the full run slow.

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

Default output locations:

| Location | Contents |
|----------|----------|
| `report/evaluation/` | Metrics, validation searches, predictions, error summaries, importance tables, run manifest, and CV outputs |
| `report/figures/` | Comparison plots, confusion matrices, ROC curves, feature importances, and objective curves |

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
