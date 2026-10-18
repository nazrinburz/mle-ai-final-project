# Contribution Report

This report summarizes the main responsibilities and approximate effort
distribution for the final project.

## Team

| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Garayzade, Gunel | [@gunelrafig](https://github.com/gunelrafig) | Evaluation orchestration, metrics, experiments, and result analysis |
| Afandiyeva, Jale | [@zhafandiyeva](https://github.com/zhafandiyeva) | SVM and linear-baseline work, including Pegasos and SVR comparisons |
| Gorchuyeva, Kheyransa | [@Kheyransa](https://github.com/Kheyransa) | Decision-tree and random-forest implementation and testing |
| Burziyeva, Nazrin | [@nazrinburziyeva](https://github.com/nazrinburziyeva) | Data preparation, integration, reproducibility, report, and presentation |

## Per-member summary

**Garayzade, Gunel --** Worked on the evaluation workflow, experiment
organization, metrics, and interpretation of model results. Helped compare
the custom implementations with the scikit-learn baselines and analyze where
the models performed well or poorly.

**Afandiyeva, Jale --** Worked on the SVM and linear-baseline experiments,
including the Pegasos SVM, Random Fourier Feature extension, Ridge regression,
and Pegasos SVR comparisons. Contributed to debugging and interpreting the
classification and regression results.

**Gorchuyeva, Kheyransa --** Implemented and tested the decision-tree and
random-forest models. Contributed to model selection, tree-based evaluation,
feature-importance analysis, and comparison with the corresponding
scikit-learn models.

**Burziyeva, Nazrin --** Implemented the shared data-preparation and integration
work, including leakage-column removal, cleaning, reproducible splitting, and
the final `run_all.py` pipeline. Completed the raw-price and log-price
evaluation integration, generated the final report and figures, prepared the
presentation, and coordinated reproducibility and documentation.

## Approximate effort split

The split below is approximate and reflects the breadth of implementation,
integration, evaluation, and documentation work rather than commit count
alone. Integration and report/presentation work required coordination across
all model components.

| Member | Approx. % |
|--------|-----------|
| Gunel Garayzade | 25% |
| Jale Afandiyeva | 25% |
| Kheyransa Gorchuyeva | 25% |
| Nazrin Burziyeva | 25% |