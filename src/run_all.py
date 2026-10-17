"""Evaluation pipeline for core and bonus models.

Run from repository root: python -m src.run_all. Default: full dataset.
Use --max-rows 400 --iterations 1000 for a clearly labelled smoke check.
Use --max-rows 0 for all rows; the team's exhaustive trees can be very slow.
Results are conditional on this protocol, not automatically final report results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor as SkForest
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.metrics import roc_curve


def regression_metrics(y_true, y_pred):
    from . import evaluate
    return {"RMSE": evaluate.rmse(y_true, y_pred), "MAE": evaluate.mae(y_true, y_pred), "R2": evaluate.r2(y_true, y_pred)}


def classification_metrics(y_true, y_pred, scores):
    from . import evaluate
    precision, recall, f1 = evaluate.precision_recall_f1(y_true, y_pred)
    return {"Precision": precision, "Recall": recall, "F1": f1, "ROC-AUC": evaluate.roc_auc(y_true, scores)}


def prepare_splits(train, val, test):
    """Fit all statistical cleaning/encoding on this training frame only."""
    from . import data_prep as dp
    threshold = float(train.price.median())
    # Adapter mirrors team's parsing but fits statistical cleaning on TRAIN.
    # Validation/test outliers are retained: do not delete hard test examples.
    floor_medians = {}
    def parse_frame(frame, fit=False):
        frame = frame.copy()
        for col in ("currency_x", "currency_y"):
            if col in frame and not frame[col].dropna().astype(str).str.upper().eq("AZN").all():
                raise ValueError(f"Non-AZN currency in {col}; agree conversion first")
        for source, target in (("Sahə", "area_m2"), ("Torpaq sahəsi", "land_area_sot")):
            if source in frame:
                frame[target] = dp._number_from_text(frame[source])
        if "Otaq sayı" in frame:
            frame["Otaq sayı"] = pd.to_numeric(frame["Otaq sayı"], errors="coerce")
        if "Mərtəbə" in frame:
            floors = frame["Mərtəbə"].astype("string").str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
            apartment = frame.get("Kateqoriya", pd.Series("", index=frame.index)).isin(dp.APARTMENT_CATEGORIES)
            for i, col in enumerate(("floor", "total_floors")):
                frame[col] = pd.to_numeric(floors[i], errors="coerce")
                frame[col + "_missing"] = frame[col].isna().astype(float)
                if "Kateqoriya" in frame:
                    if fit:
                        floor_medians[col] = frame.loc[apartment].groupby("Kateqoriya")[col].median()
                    frame.loc[apartment, col] = frame.loc[apartment, col].fillna(
                        frame.loc[apartment, "Kateqoriya"].map(floor_medians[col]))
                frame.loc[~apartment, col] = frame.loc[~apartment, col].fillna(0)
        frame = frame.drop(columns=dp.LEAKAGE_COLUMNS + dp.NON_FEATURE_COLUMNS, errors="ignore")
        # Listing promotion status may reflect price/exposure; exclude conservatively.
        frame = frame.drop(columns=["vip", "featured", "products_label"], errors="ignore")
        return frame.rename(columns=dp.AZERBAIJANI_RENAME)

    # Learn the team's IQR rule only on training prices; trim TRAIN only.
    q1, q3 = train.price.quantile([.25, .75])
    low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
    if q3 > q1:
        train = train.loc[train.price.between(low, high)].copy()
    # Threshold remains the median of the original training split, before trimming.
    tr = parse_frame(train, fit=True)
    va, te = parse_frame(val), parse_frame(test)
    # Freeze category vocabulary to prevent binary drop_first switching reference
    # when a validation/test split lacks one training category.
    for col in tr.select_dtypes(include=["object", "string", "category", "bool"]).columns:
        vocabulary = sorted(tr[col].dropna().unique().tolist(), key=str)
        for frame in (tr, va, te):
            frame[col] = pd.Categorical(frame[col], categories=vocabulary)
    processor, Xtr, ytr = dp.fit_feature_processor(tr)
    Xva, yva = dp.transform_features(va, processor)
    Xte, yte = dp.transform_features(te, processor)
    names = Xtr.columns.tolist()
    Xtr, Xva, Xte = [np.asarray(x, dtype=float) for x in (Xtr, Xva, Xte)]
    ytr, yva, yte = [np.asarray(y, dtype=float) for y in (ytr, yva, yte)]
    if not all(np.isfinite(x).all() for x in (Xtr, Xva, Xte)):
        raise ValueError("Non-finite encoded features; inspect training imputation")
    btr, bva, bte = [(y > threshold).astype(int) for y in (ytr, yva, yte)]
    return Xtr, Xva, Xte, ytr, yva, yte, btr, bva, bte, names, te, low, high, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "house_sale.csv")
    parser.add_argument("--output", type=Path, default=root / "report" / "evaluation")
    parser.add_argument("--demo", action="store_true", help="Run synthetic metric examples only")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--complete-existing", action="store_true", help="Complete saved results without retraining custom models")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV on a new full run")
    parser.add_argument("--cv-only", action="store_true", help="Evaluate saved fixed configurations on training-only CV; no test reuse")
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    if args.max_rows < 0 or args.iterations < 1:
        parser.error("max-rows must be non-negative and iterations positive")
    from . import data_prep as dp, evaluate as ev
    from .decision_tree import DecisionTree
    from .svm import PegasosSVM
    from .linear_baseline import RidgeRegression, PegasosSVR
    from .random_forest import RandomForestRegressor

    if args.demo:
        ev.compare({"Synthetic regression example": regression_metrics([10,20,30], [12,18,33])})
        ev.compare({"Synthetic classification example": classification_metrics([0,0,1,1], [0,1,0,1], [.1,.4,.35,.8])})
        return

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed = 42
    raw = dp.load_raw(str(args.data))
    original_rows = len(raw)
    # Repeated snapshots of the same listing must not span train/test.
    listing_key = "estate_rel_url_x" if "estate_rel_url_x" in raw else None
    repeated_listings = 0
    if listing_key:
        known = raw[listing_key].notna() & raw[listing_key].astype(str).str.strip().ne("")
        repeated = known & raw.duplicated(subset=[listing_key], keep="first")
        repeated_listings = int(repeated.sum())
        raw = raw.loc[~repeated].copy()
    raw = raw.drop_duplicates().copy()
    raw["price"] = dp._number_from_text(raw["price"])
    raw = raw.loc[np.isfinite(raw.price) & (raw.price > 0)].copy()
    # No target-derived sampling or global outlier trimming.
    if args.max_rows and len(raw) > args.max_rows:
        raw = raw.sample(n=args.max_rows, random_state=seed)
    train, hold = train_test_split(raw, test_size=.30, random_state=seed)
    threshold = float(train.price.median())
    # Only the held-out division is stratified; its labels use TRAIN median.
    val, test = train_test_split(
        hold, test_size=.5, random_state=seed,
        stratify=(hold.price > threshold).astype(int),
    )

    if args.cv_only:
        cross_validate_training(train, args)
        return
    Xtr, Xva, Xte, ytr, yva, yte, btr, bva, bte, names, te, low, high, threshold = prepare_splits(train, val, test)
    if args.complete_existing:
        complete_existing(train, Xtr, Xte, ytr, yte, args)
        return
    print(f"Rows train/val/test: {len(ytr)}/{len(yva)}/{len(yte)}; features: {len(names)}", flush=True)
    print("PILOT: sampled real data" if args.max_rows else "FULL DATA evaluation", flush=True)

    candidates = {}
    for task in ("regression", "classification"):
        for criterion in (("mse",) if task == "regression" else ("gini", "entropy")):
            for depth in (3, 6):
                for leaf in (5, 15):
                    params = dict(task=task, criterion=criterion, max_depth=depth,
                                  min_samples_leaf=leaf, random_state=seed)
                    candidates.setdefault("Tree " + task, []).append((params, lambda p=params: DecisionTree(**p)))
    for nonlinear in (False, True):
        key = "SVM RFF" if nonlinear else "SVM linear"
        for lam in (.001, .01):
            params = dict(lambda_=lam, n_iters=args.iterations, random_state=seed,
                          n_rff=100 if nonlinear else 0, gamma=.01 if nonlinear else None)
            candidates.setdefault(key, []).append((params, lambda p=params: PegasosSVM(**p)))
    for lam in (1., 10.):
        candidates.setdefault("Ridge", []).append((dict(lambda_=lam), lambda l=lam: RidgeRegression(lambda_=l)))
    for lam in (.001, .01):
        params = dict(lambda_=lam, epsilon=1000., n_iters=args.iterations, random_state=seed)
        candidates.setdefault("SVR", []).append((params, lambda p=params: PegasosSVR(**p)))
    for depth in (3, 6):
        params = dict(n_estimators=10, max_depth=depth, min_samples_leaf=5, random_state=seed)
        candidates.setdefault("Forest", []).append((params, lambda p=params: RandomForestRegressor(**p)))

    selected, tuning, rows, intervals = {}, [], [], []
    classifications = {"Tree classification", "SVM linear", "SVM RFF"}
    for name, choices in candidates.items():
        classification = name in classifications
        best_score = -np.inf
        for candidate_index, (params, factory) in enumerate(choices, 1):
            print(f"Training {name} [{candidate_index}/{len(choices)}]: {params}", flush=True)
            model = factory()
            start = perf_counter()
            model.fit(Xtr, btr if classification else ytr)
            elapsed = perf_counter() - start
            pred = model.predict(Xva)
            score = ev.precision_recall_f1(bva, pred)[2] if classification else -ev.rmse(yva, pred)
            tuning.append(dict(model=name, parameters=json.dumps(params), validation_score=score, fit_seconds=elapsed))
            pd.DataFrame(tuning).to_csv(out/"validation_search.csv", index=False)
            print(f"  completed in {elapsed:.2f}s; validation score={score:.6f}", flush=True)
            if score > best_score:
                best_score = score
                selected[name] = (model, params, elapsed)
        print(f"Selected {name}: validation {'F1' if classification else 'negative RMSE'}={best_score:.4f}", flush=True)

    # Baselines use each custom winner's selected settings, not test tuning.
    p = selected["Tree regression"][1]
    baselines = {"sklearn Tree regression": DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=seed)}
    p = selected["Tree classification"][1]
    baselines["sklearn Tree classification"] = DecisionTreeClassifier(criterion=p["criterion"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=seed)
    baselines["sklearn Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=selected["Ridge"][1]["lambda_"]))
    p = selected["Forest"][1]
    baselines["sklearn Forest"] = SkForest(**p, max_features="sqrt")
    for label, kernel in (("linear", "linear"), ("RFF", "rbf")):
        p = selected["SVM " + label][1]
        # Same regularization convention; exact RBF differs from approximate RFF.
        baselines["sklearn SVM " + ("RBF" if label == "RFF" else label)] = make_pipeline(StandardScaler(), SVC(
            C=1/(len(ytr)*p["lambda_"]), kernel=kernel, gamma=.01, random_state=seed))
        classifications.add("sklearn SVM " + ("RBF" if label == "RFF" else label))
    classifications.add("sklearn Tree classification")
    baselines["sklearn SVR"] = make_svr_baseline(ytr, selected["SVR"][1])
    for name, model in baselines.items():
        start = perf_counter()
        model.fit(Xtr, btr if name in classifications else ytr)
        selected[name] = (model, {}, perf_counter()-start)

    roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
    for name, (model, params, fit_time) in selected.items():
        classification = name in classifications
        start = perf_counter()
        pred = model.predict(Xte)
        predict_time = perf_counter()-start
        record = dict(Model=name, Task="B" if classification else "A", Fit_seconds=fit_time, Predict_seconds=predict_time)
        slug = name.replace(" ", "_")
        if classification:
            score = model.decision_function(Xte) if hasattr(model, "decision_function") else model.predict_proba(Xte)[:, 1]
            precision, recall, f1 = ev.precision_recall_f1(bte, pred)
            record.update(Precision=precision, Recall=recall, F1=f1, ROC_AUC=ev.roc_auc(bte, score))
            cm = ev.confusion_matrix(bte, pred)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center")
            ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["standard","premium"], yticklabels=["standard","premium"], xlabel="Predicted", ylabel="True", title=name)
            fig.tight_layout(); fig.savefig(out/f"confusion_{slug}.png", dpi=140); plt.close(fig)
            fpr, tpr, _ = roc_curve(bte, score)
            roc_ax.plot(fpr, tpr, label=f"{name}: {record['ROC_AUC']:.3f}")
            pd.DataFrame(dict(row_id=test.index, true=bte, predicted=pred, premium_score=score)).to_csv(out/f"predictions_{slug}.csv", index=False)
        else:
            record.update(RMSE=ev.rmse(yte,pred), MAE=ev.mae(yte,pred), R2=ev.r2(yte,pred))
            errors = pd.DataFrame(dict(row_id=test.index, true=yte, predicted=pred, absolute_error=np.abs(yte-pred)))
            for col in ("location", "city", "category"):
                if col in te:
                    errors[col] = te[col].astype(str).to_numpy()
                    errors.groupby(col, observed=True).absolute_error.agg(["count","mean"]).to_csv(out/f"errors_{slug}_{col}.csv")
            errors.sort_values("absolute_error", ascending=False).to_csv(out/f"errors_{slug}.csv", index=False)
        # Paired bootstrap: true values and predictions are resampled together.
        # Conditional uncertainty for this fitted model/test split, not CV std.
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(500):
            indices = rng.integers(0, len(yte), len(yte))
            if classification:
                if np.unique(bte[indices]).size < 2:
                    continue
                metrics = classification_metrics(bte[indices], pred[indices], score[indices])
            else:
                metrics = regression_metrics(yte[indices], pred[indices])
            samples.append(metrics)
        for metric in samples[0]:
            lower, upper = np.quantile([sample[metric] for sample in samples], [.025, .975])
            intervals.append(dict(model=name, metric=metric, lower_95=lower, upper_95=upper,
                                  bootstrap_samples=len(samples), method="paired test bootstrap; conditional on fitted model"))
        rows.append(record)
        if hasattr(model,"feature_importances_"):
            pd.DataFrame(dict(feature=names, importance=model.feature_importances_)).sort_values("importance",ascending=False).to_csv(out/f"importance_{slug}.csv",index=False)
        if name == "SVM linear":
            pd.DataFrame(dict(feature=names, weight=model.w)).to_csv(out/"svm_standardized_weights.csv",index=False)
        if hasattr(model,"objective_history"):
            history = np.asarray(model.objective_history)
            fig, ax = plt.subplots()
            ax.plot(history[:,0],history[:,1]); ax.set(xlabel="Iteration",ylabel="Objective",title=name)
            fig.tight_layout(); fig.savefig(out/f"objective_{slug}.png",dpi=140); plt.close(fig)
    roc_ax.plot([0,1],[0,1],"k--"); roc_ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Held-out ROC curves")
    roc_ax.legend(fontsize=7); roc_fig.tight_layout(); roc_fig.savefig(out/"roc.png",dpi=140); plt.close(roc_fig)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(intervals).to_csv(out/"metric_intervals.csv",index=False)
    pd.DataFrame(tuning).to_csv(out/"validation_search.csv",index=False)
    manifest = dict(seed=seed, raw_rows=original_rows, duplicate_listing_rows_removed=repeated_listings, sampled_rows=len(raw), max_rows=args.max_rows,
                    train_rows=len(ytr), validation_rows=len(yva), test_rows=len(yte), features=names,
                    premium_threshold=threshold, train_outlier_bounds=[low,high],
                    selected_parameters={n:p for n,(_,p,_) in selected.items()},
                    data_sha256=hashlib.file_digest(args.data.open("rb"),"sha256").hexdigest(),
                    limitations=["Pilot if max_rows > 0; not full-dataset results", "No cross-validation uncertainty yet", "Raw-price targets; outliers retained in val/test", "Local train-only cleaning adapter differs from team clean()", "RFF approximation vs exact RBF baseline", "Independent random first split; only held-out division stratified"])
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Outputs: {out}")
    analyze_saved_results(out)
    if not args.skip_cv:
        # Reconstruct original untrimmed training partition for fold fitting.
        cross_validate_training(train, args)


def cross_validate_training(training_frame, args):
    """Conditional, non-nested K-fold checks of saved fixed configurations.

    Only the original training partition is used. Previously observed test
    metrics are never used to choose parameters or filter cases.
    """
    from sklearn.model_selection import KFold
    from .decision_tree import DecisionTree
    from .svm import PegasosSVM
    from .linear_baseline import RidgeRegression, PegasosSVR
    from .random_forest import RandomForestRegressor
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    if args.max_rows != manifest["max_rows"]:
        raise ValueError("CV sampling must match the saved run")
    if hashlib.file_digest(args.data.open("rb"), "sha256").hexdigest() != manifest["data_sha256"]:
        raise ValueError("Dataset differs from the saved evaluation")
    factories = {"Tree regression": DecisionTree, "Tree classification": DecisionTree,
                 "SVM linear": PegasosSVM, "SVM RFF": PegasosSVM,
                 "Ridge": RidgeRegression, "SVR": PegasosSVR, "Forest": RandomForestRegressor}
    classifications = {"Tree classification", "SVM linear", "SVM RFF", "sklearn Tree classification", "sklearn SVM linear", "sklearn SVM RBF"}
    rows = []
    splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    for fold, (itr, iva) in enumerate(splitter.split(training_frame), 1):
        foldtrain, foldval = training_frame.iloc[itr], training_frame.iloc[iva]
        Xtr, Xva, _, ytr, yva, _, btr, bva, _, _, _, _, _, _ = prepare_splits(foldtrain, foldval, foldval)
        models = {name: factory(**manifest["selected_parameters"][name]) for name, factory in factories.items()}
        p = manifest["selected_parameters"]["Tree regression"]
        models["sklearn Tree regression"] = DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=42)
        p = manifest["selected_parameters"]["Tree classification"]
        models["sklearn Tree classification"] = DecisionTreeClassifier(criterion=p["criterion"],max_depth=p["max_depth"],min_samples_leaf=p["min_samples_leaf"],random_state=42)
        models["sklearn Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=manifest["selected_parameters"]["Ridge"]["lambda_"]))
        models["sklearn Forest"] = SkForest(**manifest["selected_parameters"]["Forest"], max_features="sqrt")
        for label, kernel in (("linear", "linear"), ("RFF", "rbf")):
            p = manifest["selected_parameters"]["SVM " + label]
            name = "sklearn SVM " + ("RBF" if label == "RFF" else label)
            models[name] = make_pipeline(StandardScaler(), SVC(C=1/(len(ytr)*p["lambda_"]), kernel=kernel,gamma=.01,random_state=42))
        models["sklearn SVR"] = make_svr_baseline(ytr, manifest["selected_parameters"]["SVR"])
        for name, model in models.items():
            print(f"CV fold {fold}/{args.cv_folds}: training {name}", flush=True)
            classification = name in classifications
            begin = perf_counter()
            model.fit(Xtr, btr if classification else ytr)
            fit_seconds = perf_counter()-begin
            pred = model.predict(Xva)
            if classification:
                scores = model.decision_function(Xva) if hasattr(model,"decision_function") else model.predict_proba(Xva)[:,1]
                metrics = classification_metrics(bva,pred,scores)
            else:
                metrics = regression_metrics(yva,pred)
            rows.append(dict(fold=fold, model=name, fit_seconds=fit_seconds, **metrics))
            table = pd.DataFrame(rows)
            table.to_csv(args.output/"cv_folds.csv",index=False)
            print(f"  {fit_seconds:.2f}s: {metrics}",flush=True)
    metric_columns = [c for c in table if c not in ("fold","model")]
    summary = table.groupby("model")[metric_columns].agg(["mean","std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(args.output/"cv_summary.csv")
    (args.output/"cv_protocol.json").write_text(json.dumps(dict(
        folds=args.cv_folds, seed=42, scope="Original training partition only; training-fitted preprocessing in each fold",
        selection="Fixed configurations previously selected on validation; non-nested, conditional CV, not an unbiased nested selection estimate",
        tier="Training-fold price median; outliers removed from fit fold only", final_test="Not accessed in CV scoring"),indent=2),encoding="utf-8")
    print("Training-only CV complete",flush=True)


def make_svr_baseline(y_train, parameters):
    """Same standardized-X/y primal objective; exact linear kernel solver."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.svm import SVR
    y_scale = float(np.std(y_train)) or 1.0
    model = make_pipeline(StandardScaler(), SVR(
        kernel="linear", C=1/(len(y_train)*parameters["lambda_"]),
        epsilon=parameters["epsilon"]/y_scale, max_iter=200000))
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def analyze_saved_results(out):
    """Regenerate derived tables/figures from saved predictions; no training."""
    metrics = pd.read_csv(out/"metrics.csv")
    metrics.Model = metrics.Model.replace({"sklearn SVM RFF":"sklearn SVM RBF"})
    metrics.to_csv(out/"comparison_table.csv",index=False)
    classification_rows, regression_rows = [], []
    roc_fig, roc_ax = plt.subplots(figsize=(8,5))
    for path in sorted(out.glob("predictions_*.csv")):
        frame = pd.read_csv(path)
        name = path.stem.removeprefix("predictions_").replace("_", " ").replace("sklearn SVM RFF", "sklearn SVM RBF")
        from . import evaluate
        tn, fp, fn, tp = evaluate.confusion_matrix(frame["true"],frame.predicted).ravel()
        classification_rows.append(dict(model=name,TN=tn,FP=fp,FN=fn,TP=tp,test_samples=len(frame)))
        fpr,tpr,_ = roc_curve(frame["true"],frame.premium_score)
        roc_ax.plot(fpr,tpr,label=name)
    roc_ax.plot([0,1],[0,1],"k--")
    roc_ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Held-out ROC curves")
    roc_ax.legend(fontsize=7);roc_fig.tight_layout();roc_fig.savefig(out/"roc_corrected_labels.png",dpi=160);plt.close(roc_fig)
    pd.DataFrame(classification_rows).to_csv(out/"classification_error_summary.csv",index=False)
    for path in sorted(out.glob("errors_*.csv")):
        frame = pd.read_csv(path)
        if "true" not in frame:
            continue
        errors = np.sort(frame.absolute_error.to_numpy())[::-1]
        regression_rows.append(dict(model=path.stem.removeprefix("errors_").replace("_"," "),
            median_absolute_error=np.median(errors),
            top_10_squared_error_percent=100*np.sum(errors[:10]**2)/np.sum(errors**2),
            max_absolute_error=errors[0]))
    pd.DataFrame(regression_rows).to_csv(out/"regression_error_summary.csv",index=False)
    for name in ("Tree_classification","Tree_regression","Forest"):
        frame = pd.read_csv(out/f"importance_{name}.csv").head(10).iloc[::-1]
        fig,ax=plt.subplots(figsize=(9,5));ax.barh(frame.feature,frame.importance)
        ax.set(xlabel="Normalized impurity-decrease importance",title=name.replace("_"," "))
        fig.tight_layout();fig.savefig(out/f"feature_importance_{name}.png",dpi=160);plt.close(fig)
    frame=pd.read_csv(out/"svm_standardized_weights.csv")
    frame["absolute_weight"]=frame.weight.abs()
    frame=frame.sort_values("absolute_weight",ascending=False)
    frame.to_csv(out/"svm_weights_ranked.csv",index=False)
    top=frame.head(10).iloc[::-1];fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(top.feature,top.weight,color=["tab:blue" if v>=0 else "tab:red" for v in top.weight])
    ax.set(xlabel="Weight in standardized feature space",title="Linear SVM: largest absolute weights")
    fig.tight_layout();fig.savefig(out/"svm_weights.png",dpi=160);plt.close(fig)


def complete_existing(training_frame, Xtr, Xte, ytr, yte, args):
    """Fill missing SVR baseline and derived files, preserving custom results."""
    out=args.output
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if args.max_rows != manifest["max_rows"] or hashlib.file_digest(args.data.open("rb"),"sha256").hexdigest()!=manifest["data_sha256"]:
        raise ValueError("Saved run dataset/sampling mismatch")
    if len(ytr)!=manifest["train_rows"] or len(yte)!=manifest["test_rows"]:
        raise ValueError("Saved split differs; do not mix results")
    rows=pd.read_csv(out/"metrics.csv")
    intervals=pd.read_csv(out/"metric_intervals.csv")
    def fit_baseline(X,y,Xv,yv):
        model=make_svr_baseline(y,manifest["selected_parameters"]["SVR"])
        print("Training missing sklearn linear SVR baseline",flush=True)
        begin=perf_counter();model.fit(X,y);fit=perf_counter()-begin
        begin=perf_counter();pred=model.predict(Xv);predict=perf_counter()-begin
        print(f"SVR baseline complete: {fit:.2f}s",flush=True)
        return pred,dict(Fit_seconds=fit,Predict_seconds=predict,**regression_metrics(yv,pred))
    if "sklearn SVR" not in rows.Model.values:
        pred,values=fit_baseline(Xtr,ytr,Xte,yte)
        rows=pd.concat([rows,pd.DataFrame([dict(Model="sklearn SVR",Task="A",**values)])],ignore_index=True)
        rows.to_csv(out/"metrics.csv",index=False)
        error_frame=pd.DataFrame(dict(true=yte,predicted=pred,absolute_error=np.abs(yte-pred)))
        error_frame.sort_values("absolute_error",ascending=False).to_csv(out/"errors_sklearn_SVR.csv",index=False)
        rng=np.random.default_rng(42);boot=[]
        for _ in range(500):
            idx=rng.integers(0,len(yte),len(yte));boot.append(regression_metrics(yte[idx],pred[idx]))
        extra=[]
        for metric in boot[0]:
            lower,upper=np.quantile([r[metric] for r in boot],[.025,.975])
            extra.append(dict(model="sklearn SVR",metric=metric,lower_95=lower,upper_95=upper,bootstrap_samples=500,method="paired test bootstrap; conditional on fitted model"))
        pd.concat([intervals,pd.DataFrame(extra)],ignore_index=True).to_csv(out/"metric_intervals.csv",index=False)
    folds=pd.read_csv(out/"cv_folds.csv")
    if "sklearn SVR" not in folds.model.values:
        from sklearn.model_selection import KFold
        extra=[]
        for fold,(itr,iva) in enumerate(KFold(n_splits=3,shuffle=True,random_state=42).split(training_frame),1):
            tr,va=training_frame.iloc[itr],training_frame.iloc[iva]
            X,yX,_,y,yv,*_=prepare_splits(tr,va,va)
            _,values=fit_baseline(X,y,yX,yv)
            extra.append(dict(fold=fold,model="sklearn SVR",fit_seconds=values.pop("Fit_seconds"),**{k:v for k,v in values.items() if k!="Predict_seconds"}))
        folds=pd.concat([folds,pd.DataFrame(extra)],ignore_index=True)
        folds.to_csv(out/"cv_folds.csv",index=False)
        cols=[c for c in folds if c not in ("fold","model")]
        summary=folds.groupby("model")[cols].agg(["mean","std"])
        summary.columns=["_".join(c) for c in summary.columns]
        summary.to_csv(out/"cv_summary.csv")
    manifest["limitations"]=[x for x in manifest["limitations"] if x not in ("No sklearn SVR matched baseline yet","No cross-validation uncertainty yet")]
    manifest["cv_protocol_file"]="cv_protocol.json"
    manifest["protocol_status"]="Training-only IQR protocol pending team agreement; results conditional on this choice"
    manifest["selected_parameters"]["sklearn SVR"]={"kernel":"linear","C_convention":"1/(n_train*lambda)","target_scaling":"training mean/std","epsilon":"original epsilon / training target std"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    analyze_saved_results(out)
    print("Saved-result completion finished",flush=True)


if __name__ == "__main__":
    main()
