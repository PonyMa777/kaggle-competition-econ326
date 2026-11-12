"""Compare the predeclared A/B models using identical five-fold splits.
Run from the repository root: python scripts/cross_validate_models.py
"""
from __future__ import annotations
import argparse
import hashlib
import platform
import numpy as np
import pandas as pd
from fit_gradient_descent import (
    ROOT, FEATURES, prepare_features, prepare_training_features,
    fit_gradient_descent, write_json,
)

MODELS = {
    "A_basic": ["GrLivArea", "OverallQual", "Age"],
    "B_full": FEATURES,
}
BASELINE = "Mean_price_baseline"

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual-predicted)**2)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_path = ROOT/"data"/"house-prices-advanced-regression-techniques"/"train.csv"
    output = ROOT/"results"/"five_fold_cv"
    data = pd.read_csv(data_path)
    n = len(data)
    if n < 5 or data["Id"].duplicated().any():
        raise ValueError("Need at least five rows with unique IDs.")
    y = data["SalePrice"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite outcome.")
    # Set membership before computing any validation errors; same split for all models.
    rng = np.random.default_rng(args.seed)
    validation_folds = np.array_split(rng.permutation(n), 5)
    coverage, fold_numbers = np.zeros(n, int), np.zeros(n, int)
    names = [BASELINE]+list(MODELS)
    oof = {name: np.full(n, np.nan) for name in names}
    rows, details = [], []
    for fold, validation_indices in enumerate(validation_folds, 1):
        mask = np.ones(n, dtype=bool)
        mask[validation_indices] = False
        train_indices = np.flatnonzero(mask)
        assert not np.intersect1d(train_indices, validation_indices).size
        training = data.iloc[train_indices].copy()
        validation = data.iloc[validation_indices].drop(columns=["SalePrice"]).copy()
        X_train_all, y_train, train_audit = prepare_training_features(training)
        X_valid_all, valid_audit = prepare_features(
            validation, training_rules=train_audit["categorical_rules"])
        y_valid = y[validation_indices]
        coverage[validation_indices] += 1
        fold_numbers[validation_indices] = fold
        baseline = float(y_train.mean())
        oof[BASELINE][validation_indices] = baseline
        rows.append({
            "model": BASELINE, "fold": fold, "feature_count": 0,
            "train_n": len(y_train), "validation_n": len(y_valid),
            "training_rmse_usd": rmse(y_train, np.full(len(y_train), baseline)),
            "validation_rmse_usd": rmse(y_valid, np.full(len(y_valid), baseline)),
            "iterations": 0, "learning_rate": None,
            "max_validation_prediction_difference_vs_lstsq_usd": None,
        })
        detail = {"fold": fold, "training_preprocessing": train_audit,
                  "validation_preprocessing_audit": valid_audit, "models": {}}
        for name, features in MODELS.items():
            X_train, X_valid = (X_train_all[features].to_numpy(),
                                X_valid_all[features].to_numpy())
            # Learn all scaling parameters exclusively from training rows.
            means, scales = X_train.mean(axis=0), X_train.std(axis=0, ddof=0)
            if (scales == 0).any():
                raise ValueError(f"{name}, fold {fold}: constant training feature.")
            A_train = np.column_stack([np.ones(len(y_train)), (X_train-means)/scales])
            A_valid = np.column_stack([np.ones(len(y_valid)), (X_valid-means)/scales])
            theta, history, optimization = fit_gradient_descent(
                A_train, y_train, alpha=None, tolerance=1e-8, max_iterations=500000)
            train_predictions, valid_predictions = A_train@theta, A_valid@theta
            # Only AFTER GD: independent least-squares verification, not GD training.
            reference, _, rank, _ = np.linalg.lstsq(A_train, y_train, rcond=None)
            difference = float(np.max(np.abs(A_valid@reference-valid_predictions)))
            gradient_norm = float(np.linalg.norm(
                A_train.T@(train_predictions-y_train)/len(y_train)))
            raw_weights = theta[1:]/scales
            raw_intercept = float(theta[0]-means@raw_weights)
            if rank != len(features)+1 or difference >= .001 or gradient_norm > 1e-8:
                raise AssertionError(f"{name}, fold {fold}: optimizer check failed.")
            if not np.allclose(raw_intercept+X_valid@raw_weights, valid_predictions,
                               atol=1e-6, rtol=0):
                raise AssertionError("Unscaling failed.")
            if not np.all(np.diff(history["cost_usd2"]) <= 1e-4):
                raise AssertionError("Cost increased beyond roundoff.")
            oof[name][validation_indices] = valid_predictions
            row = {
                "model": name, "fold": fold, "feature_count": len(features),
                "train_n": len(y_train), "validation_n": len(y_valid),
                "training_rmse_usd": rmse(y_train, train_predictions),
                "validation_rmse_usd": rmse(y_valid, valid_predictions),
                "iterations": optimization["iterations"],
                "learning_rate": optimization["learning_rate"],
                "max_validation_prediction_difference_vs_lstsq_usd": difference,
            }
            rows.append(row)
            detail["models"][name] = {
                "feature_order": features, "means": means.tolist(),
                "standard_deviations_ddof0": scales.tolist(),
                "standardized_intercept": float(theta[0]),
                "standardized_weights": theta[1:].tolist(),
                "raw_intercept": raw_intercept, "raw_weights": raw_weights.tolist(),
                "optimization": optimization,
                "max_validation_prediction_difference_vs_lstsq_usd": difference,
                "direct_gradient_l2_norm": gradient_norm,
            }
            print(f"Fold {fold}, {name}: validation RMSE "
                  f"USD {row['validation_rmse_usd']:,.2f}", flush=True)
        details.append(detail)
    if not np.all(coverage == 1):
        raise AssertionError("Every house must be validated exactly once.")
    if any(not np.isfinite(pred).all() for pred in oof.values()):
        raise AssertionError("Missing/nonfinite out-of-fold prediction.")
    folds = pd.DataFrame(rows)
    summaries = []
    for name in names:
        group = folds.loc[folds["model"].eq(name)]
        errors = group["validation_rmse_usd"].to_numpy()
        pooled = rmse(y, oof[name])
        if abs(pooled-np.sqrt(np.average(errors**2, weights=group["validation_n"]))) > 1e-7:
            raise AssertionError("Fold metrics and pooled metrics do not reconcile.")
        summaries.append({
            "model": name, "feature_count": int(group["feature_count"].iloc[0]),
            "mean_validation_rmse_usd": float(errors.mean()),
            "sd_validation_rmse_usd": float(errors.std(ddof=1)),
            "pooled_oof_rmse_usd": pooled,
            "mean_training_rmse_usd": float(group["training_rmse_usd"].mean()),
        })
    comparison = pd.DataFrame(summaries).sort_values("mean_validation_rmse_usd")
    candidates = comparison[comparison["model"].isin(MODELS)]
    winner = str(candidates.iloc[0]["model"])
    indexed = comparison.set_index("model")
    basic = float(indexed.loc["A_basic", "mean_validation_rmse_usd"])
    full = float(indexed.loc["B_full", "mean_validation_rmse_usd"])
    metadata = {
        "selection_rule": "Lowest arithmetic mean of 5 validation RMSEs among A and B",
        "winner_among_compared_models": winner, "candidate_features": MODELS,
        "n_splits": 5, "seed": args.seed, "sample_count": n,
        "split_algorithm": "numpy.random.default_rng(seed).permutation; numpy.array_split",
        "target": "SalePrice in USD, no log",
        "preprocessing": "Modes, means and std(ddof=0) fitted on each training fold only",
        "optimizer": "Zero-initialized full-batch GD; alpha=1/L from training design",
        "B_reduction_in_mean_validation_rmse_vs_A_usd": basic-full,
        "B_reduction_in_mean_validation_rmse_vs_A_percent": 100*(basic-full)/basic,
        "baseline_used_for_selection": False,
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__},
        "checks": {
            "every_house_validated_once": bool(np.all(coverage == 1)),
            "training_validation_disjoint": True, "all_models_share_folds": True,
            "all_predictions_finite": True, "all_10_gd_fits_verified_against_lstsq": True,
            "outcome_excluded_from_validation_preprocessing": True,
        },
        "interpretation": "Model-selection CV, not an independent final test estimate.",
    }
    predictions = pd.DataFrame({"Id": data["Id"], "fold": fold_numbers,
                                "SalePrice": y, **oof})
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("comparison.csv", comparison), ("fold_metrics.csv", folds),
        ("fold_assignments.csv", predictions[["Id", "fold"]]),
        ("out_of_fold_predictions.csv", predictions),
    ]:
        table.to_csv(output/name, index=False, float_format="%.15g")
    write_json(output/"metadata.json", metadata)
    write_json(output/"fold_models.json", details)
    write_report(output, comparison, folds, metadata)
    print(comparison.to_string(index=False), flush=True)
    print(f"Winner: {winner}; B RMSE reduction vs A: USD {basic-full:,.2f} "
          f"({100*(basic-full)/basic:.2f}%)", flush=True)

def write_report(output, comparison, folds, metadata):
    rows = "\n".join(
        f"| {r.model} | {r.feature_count} | {r.mean_validation_rmse_usd:,.2f} | "
        f"{r.sd_validation_rmse_usd:,.2f} | {r.pooled_oof_rmse_usd:,.2f} |"
        for r in comparison.itertuples())
    pivot = folds.pivot(index="fold", columns="model", values="validation_rmse_usd")
    fold_rows = "\n".join(
        f"| {fold} | {pivot.loc[fold, BASELINE]:,.2f} | "
        f"{pivot.loc[fold, 'A_basic']:,.2f} | {pivot.loc[fold, 'B_full']:,.2f} |"
        for fold in pivot.index)
    winner = metadata["winner_among_compared_models"]
    reduction = metadata["B_reduction_in_mean_validation_rmse_vs_A_usd"]
    percent = metadata["B_reduction_in_mean_validation_rmse_vs_A_percent"]
    better_folds = int((pivot["B_full"] < pivot["A_basic"]).sum())
    worse_folds = pivot.index[pivot["B_full"] > pivot["A_basic"]].tolist()
    variability = (
        f"B has lower RMSE in {better_folds} of the five folds. "
        f"Folds where B has higher RMSE: {worse_folds}.\n"
        "Read the fold standard deviations alongside the mean errors.\n\n"
    )
    fence = chr(96)*3
    report = f"""# Five-fold cross-validation: basic versus full model

**{winner} has the lowest mean validation RMSE among the two compared models.**
The 18-feature model reduces mean RMSE by USD {reduction:,.2f}
({percent:.2f}%) relative to the basic model.

## Models specified before validation

- **A_basic:** GrLivArea, OverallQual and Age (3 features).
- **B_full:** all 18 features in [outline.md](../../outline.md), using the agreed
  numeric coding and constructed features.
- **Mean_price_baseline:** predict the training fold's mean SalePrice for every
  validation house. This is a reference, not a feature candidate.
- NearISU has not been created and is not included.

## Validation results (USD; lower is better)

| Model | Features | Mean of 5 fold RMSEs | SD across folds | Pooled OOF RMSE |
|---|---:|---:|---:|---:|
{rows}

Selection uses the **arithmetic mean of the five RMSEs**. SD is the sample
standard deviation across folds, not a confidence interval. Pooled OOF RMSE
combines all held-out squared errors before taking the square root; it differs
from mean fold RMSE. Neither is a training error.

| Fold | Mean-price baseline | A: 3 features | B: 18 features |
|---|---:|---:|---:|
{fold_rows}

## Procedure

1. Shuffle the labeled rows once with seed {metadata['seed']} and divide them into
   five folds. Every model uses the same assignments. In this 1,460-row sample,
   each fit trains on 1,168 houses and predicts the remaining 292.
2. Fit missing-category modes using training rows only. Confirmed absence receives
   score 0. Apply the fitted rules to validation rows without learning new modes.
3. Calculate each feature's mean and standard deviation on training rows only,
   and use them to standardize both subsets. SalePrice stays in dollars.
4. Start coefficients at zero. Fit by full-batch gradient descent; the learning
   rate is 1/L from the training design. Validation prices do not choose the
   learning rate or stop training.
5. Score held-out prices. Repeat until every house has exactly one held-out
   prediction, then average the five RMSEs for each model.

All ten GD fits passed convergence and independent least-squares checks.
Within each fold, training and validation IDs are disjoint. SalePrice is removed
before validation preprocessing. All predictions are finite.

## Interpretation

{variability}This identifies the better of **these two feature sets on this five-fold split**.
It does not prove that every added feature helps, or find the best of all possible
feature combinations. The CV errors also guide model selection; they are not an
independent final test evaluation. The unlabeled competition test set was not used.

Fold coefficients belong to different training subsets; do not average them to
form the final model. Fit the chosen feature set on all labeled rows.
The existing [full-data 18-feature fit](../gradient_descent/README.md) is stored
separately with its training error clearly labeled.

## Reproduce and inspect

{fence}bash
python -m pip install -r requirements.txt
python scripts/cross_validate_models.py
{fence}

- [Source](../../scripts/cross_validate_models.py)
- [Comparison](comparison.csv) and [per-fold metrics](fold_metrics.csv)
- [Exact fold assignments](fold_assignments.csv)
- [Out-of-fold predictions](out_of_fold_predictions.csv)
- [Fold preprocessing parameters and coefficients](fold_models.json)
- [Seed, input hash, and verification](metadata.json)
"""
    (output/"README.md").write_text(report, encoding="utf-8")

if __name__ == "__main__":
    main()
