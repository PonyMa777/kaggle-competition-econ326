"""Fit the current 18-feature dollar-price model using full-batch gradient descent.
Run from the repository root: python scripts/fit_gradient_descent.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "ExterCond", "BsmtCond", "BsmtExposure", "HeatingQC", "GrLivArea",
    "Fireplaces", "GarageArea", "GarageQual", "PavedDrive", "PoolQC",
    "Age", "QualityAdjustedLivingArea", "RenovationFreshness",
]
QUALITY = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
CATEGORIES = {
    "ExterCond": (QUALITY, None),
    "BsmtCond": (QUALITY, "TotalBsmtSF"),
    "BsmtExposure": ({"No": 1, "Mn": 2, "Av": 3, "Gd": 4}, "TotalBsmtSF"),
    "HeatingQC": (QUALITY, None),
    "GarageQual": (QUALITY, "GarageArea"),
    "PavedDrive": ({"N": 1, "P": 2, "Y": 3}, None),
    "PoolQC": (QUALITY, "PoolArea"),
}

def prepare_features(frame, training_rules=None):
    """Fit modes on training rows, or apply supplied modes to other rows.

    This transformation never reads or requires SalePrice.
    """
    numeric = [name for name in FEATURES[:15] if name not in CATEGORIES]
    required = numeric + list(CATEGORIES) + [
        "YrSold", "TotalBsmtSF", "PoolArea", "Id"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame["Id"].duplicated().any():
        raise ValueError("Duplicate house IDs.")
    X = frame[numeric].astype(float).copy()
    rules = {}
    for name, (mapping, area_field) in CATEGORIES.items():
        values = frame[name].copy()
        unexpected = sorted(set(values.dropna()) - set(mapping))
        if unexpected:
            raise ValueError(f"Unexpected {name} levels: {unexpected}")
        absent = (values.isna() & frame[area_field].eq(0) if area_field
                  else pd.Series(False, index=frame.index))
        unknown = values.isna() & ~absent
        if training_rules is None:
            modes = values.dropna().mode()
            if modes.empty:
                raise ValueError(f"No observed {name} level to learn a mode.")
            # Ties use the first sorted mode; unknown is not absence=0.
            mode = str(modes.iloc[0])
        else:
            mode = training_rules[name]["training_mode"]
        if mode not in mapping:
            raise ValueError(f"Invalid fitted mode for {name}: {mode}")
        scores = values.map(mapping).astype(float)
        scores.loc[absent] = 0.0
        scores.loc[unknown] = float(mapping[mode])
        X[name] = scores
        rules[name] = {
            "mapping": mapping, "absence_area_field": area_field,
            "absence_score": 0 if area_field else None,
            "confirmed_absence_count": int(absent.sum()),
            "unknown_count": int(unknown.sum()),
            "unknown_house_ids": frame.loc[unknown, "Id"].astype(int).tolist(),
            "training_mode": mode,
        }
    age = frame["YrSold"] - frame["YearBuilt"]
    remodel_age = frame["YrSold"] - frame["YearRemodAdd"]
    X["Age"] = age.clip(lower=0)
    X["QualityAdjustedLivingArea"] = frame["GrLivArea"] * frame["OverallQual"]
    X["RenovationFreshness"] = 1.0 / (1.0 + remodel_age.clip(lower=0))
    X = X[FEATURES].astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Unresolved missing/nonfinite numeric input.")
    audit = {
        "categorical_rules": rules,
        "negative_age_count": int((age < 0).sum()),
        "negative_remodel_age_count": int((remodel_age < 0).sum()),
        "negative_remodel_age_house_ids":
            frame.loc[remodel_age < 0, "Id"].astype(int).tolist(),
        "constructed_features": {
            "Age": "max(0, YrSold - YearBuilt)",
            "QualityAdjustedLivingArea": "GrLivArea * OverallQual",
            "RenovationFreshness": "1 / (1 + max(0, YrSold - YearRemodAdd))",
        },
    }
    return X, audit

def prepare_training_features(frame):
    """Fit preprocessing and read the labeled training outcome."""
    X, audit = prepare_features(frame)
    y = frame["SalePrice"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Missing/nonfinite training outcome.")
    return X, y, audit

def cost(A, y, theta):
    error = A @ theta - y
    return float(error @ error / (2.0 * len(y)))

def fit_gradient_descent(A, y, alpha, tolerance, max_iterations):
    """theta[0]=b, theta[1:]=w; simultaneous full-batch gradient updates."""
    m = len(y)
    # A contains an intercept column followed by standardized features.
    # dJ/dtheta = A.T @ (A @ theta - y) / m
    #           = (A.T @ A / m) @ theta - A.T @ y / m.
    # Cache fixed products for speed. This is the SAME batch GD gradient;
    # no inversion or normal-equation solution is used to obtain theta.
    gram, rhs = A.T @ A / m, A.T @ y / m
    eigenvalues = np.linalg.eigvalsh(gram)
    lower, upper = float(eigenvalues[0]), float(eigenvalues[-1])
    if lower <= 1e-12:
        raise ValueError("Design is not full rank; unique coefficients not assured.")
    if alpha is None:
        alpha = 1.0 / upper
    if not (0 < alpha <= (1.0 / upper) * (1 + 1e-12)):
        raise ValueError(f"Use 0 < learning rate <= {1.0 / upper:.12g}.")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("Tolerance and maximum iterations must be positive.")
    theta = np.zeros(A.shape[1])
    history, converged = [], False
    for iteration in range(max_iterations + 1):
        gradient = gram @ theta - rhs
        norm = float(np.linalg.norm(gradient))
        if norm <= tolerance:
            # Independently check the residual-based gradient before stopping.
            gradient = A.T @ (A @ theta - y) / m
            norm = float(np.linalg.norm(gradient))
            converged = norm <= tolerance
        if iteration <= 10 or iteration % 500 == 0 or converged:
            current_cost = cost(A, y, theta)
            history.append({
                "iteration": iteration, "cost_usd2": current_cost,
                "training_rmse_usd": float(np.sqrt(2 * current_cost)),
                "gradient_l2_norm": norm,
            })
        if converged:
            break
        if iteration == max_iterations:
            raise RuntimeError("GD did not converge; fitted results not saved.")
        theta = theta - alpha * gradient  # All RHS values use OLD theta.
    return theta, pd.DataFrame(history), {
        "converged": converged, "iterations": iteration, "learning_rate": alpha,
        "gradient_l2_tolerance": tolerance, "gradient_l2_norm": norm,
        "smallest_hessian_eigenvalue": lower,
        "largest_hessian_eigenvalue": upper,
        "hessian_condition_number": upper / lower,
        "initialization": "all weights and intercept = 0",
        "history_sampling": "iterations 0-10, every 500, and final iteration",
    }

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")

def write_report(output, metrics, coefficients, raw_weights, audit):
    opt, check = metrics["optimization"], metrics["verification"]
    rows = "\n".join(
        f"| {r.parameter} | {r.feature} | {r.raw_input_coefficient:,.6f} |"
        for r in coefficients.itertuples())
    rules = audit["categorical_rules"]
    missing_rows = "\n".join(
        f"| {name} | {v['confirmed_absence_count']} | {v['unknown_count']} | "
        f"{v['training_mode']} |" for name, v in rules.items())
    fence = chr(96) * 3
    report = f"""# Gradient-descent fit: current 18-feature model

**Training complete.** Full-batch gradient descent reached the minimum training
cost to numerical precision for the current 18-feature model, using all
{metrics['sample_count']:,} labeled houses. The target is SalePrice in US dollars.

## Results

| Quantity | Value |
|---|---:|
| Training cost J = MSE / 2 (USD squared) | {metrics['training_cost_usd2']:,.6f} |
| Training RMSE (USD) | {metrics['training_rmse_usd']:,.6f} |
| Training R-squared | {metrics['training_r_squared']:.6f} |
| Initial cost (all coefficients zero) | {metrics['initial_cost_usd2']:,.6f} |
| Mean-price baseline cost | {metrics['mean_prediction_baseline_cost_usd2']:,.6f} |
| Learning rate | {opt['learning_rate']:.12f} |
| Iterations | {opt['iterations']:,} |

These are **training results**, not out-of-sample accuracy. The separate
[five-fold comparison](../five_fold_cv/README.md) reports validation performance.
NearISU has not been constructed and is not included in this fit.

## Coefficients for the original feature units

Apply the coding and construction in [outline.md](../../outline.md), multiply
each unscaled feature by its matching coefficient, and add the intercept:

$$
\\hat y_i = b + w_1\\mathrm{{LotArea}}_i + w_2\\mathrm{{OverallQual}}_i
+ \\cdots + w_{{18}}\\mathrm{{RenovationFreshness}}_i.
$$

| Parameter | Feature | Coefficient (original feature units) |
|---|---|---:|
{rows}

Values above are rounded. [model.json](model.json) stores full precision;
[coefficients.csv](coefficients.csv) gives both raw-input and standardized-input
coefficients. Both sets produce the same dollar predictions. Do not mix the scales.

A negative coefficient is not automatically an error or a causal finding.
The model includes GrLivArea and GrLivArea × OverallQual, so its slope with
respect to living area, holding other underlying attributes fixed, is
**{raw_weights[9]:.6f} + {raw_weights[16]:.6f} × OverallQual**.
Age and YearBuilt are highly correlated; their separate coefficients also need
careful interpretation. No positive-sign restrictions were imposed.

## Preparation and fitting

All rows and all 18 selected features were kept. SalePrice was not logged.
Only a documented absent facility receives score 0. For an unknown rating, the
mode is learned from observed training categories.

| Field | Confirmed absence: score 0 | Unknown entries imputed | Training mode |
|---|---:|---:|---|
{missing_rows}

Negative ages were capped at zero: {audit['negative_age_count']} construction-age
records and {audit['negative_remodel_age_count']} remodeling-age records.
The three constructed features follow outline.md exactly.

Each feature was standardized: z_ij = (x_ij - mean_j) / std_j, using ddof=0.
Imputation and scaling were fitted on this training sample only. In future
cross-validation, they must be learned again within each training fold.

Starting from b = 0 and all weights = 0, we used simultaneous updates:

$$
J(\\vec w,b)=\\frac{{1}}{{m}}\\sum_{{i=1}}^m
\\frac{{1}}{{2}}(\\vec w\\cdot\\vec z_i+b-y_i)^2.
$$

$$
\\frac{{\\partial J}}{{\\partial w_j}}=\\frac{{1}}{{m}}
\\sum_{{i=1}}^m(\\hat y_i-y_i)z_{{ij}},
\\qquad
\\frac{{\\partial J}}{{\\partial b}}=\\frac{{1}}{{m}}\\sum_{{i=1}}^m(\\hat y_i-y_i).
$$

$$
w_j^{{new}}=w_j^{{old}}-\\alpha\\frac{{\\partial J}}{{\\partial w_j}},
\\qquad b^{{new}}=b^{{old}}-\\alpha\\frac{{\\partial J}}{{\\partial b}}.
$$

We go through w1 to w18, calculate each derivative at the old coefficients, and
then update all weights and b together. Fixed matrix products in this same
gradient are cached to make the iterations faster.

The learning rate is 1/L, where L is the largest eigenvalue of this quadratic
cost's Hessian after standardization. We stop when the full gradient norm is at
most {opt['gradient_l2_tolerance']:g}; small cost changes alone do not stop training.
There is no regularization.

Convert back to original units using w_raw,j = w_scaled,j / std_j and
b_raw = b_scaled - sum_j(w_raw,j × mean_j). Scaling changes the units used during
optimization, not the set of possible fitted linear models.

## Verification

- The design including the intercept has full rank
  {check['design_rank_including_intercept']}; this quadratic cost has a unique
  minimizer for the fixed data and features.
- An independent least-squares calculation was run **after** GD, only to verify
  the result. Its largest fitted-price difference from GD is
  **{check['max_prediction_difference_vs_lstsq_usd']:.10f} USD**.
- Raw-input and standardized-input predictions differ by at most
  **{check['max_prediction_difference_after_unscaling_usd']:.10f} USD**.
- The direct residual-based gradient norm is
  **{check['direct_gradient_l2_norm']:.3e}**.
- Recorded cost is nonincreasing up to floating-point roundoff.
- These checks establish optimization accuracy, not out-of-sample performance.

## Reproduce

From the repository root (Python 3.12 was used):

{fence}bash
python -m pip install -r requirements.txt
python scripts/fit_gradient_descent.py
{fence}

[Source code](../../scripts/fit_gradient_descent.py) implements preprocessing,
GD, checks and output generation.
[metrics.json](metrics.json) records settings, checks, software versions and the
input-file hash. [feature_scaling.csv](feature_scaling.csv) records means and
standard deviations. [convergence.csv](convergence.csv) records iterations 0-10,
every 500, and the last. [training_predictions.csv](training_predictions.csv)
contains fitted training prices and residuals, not competition-test predictions.
"""
    (output / "README.md").write_text(report, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" /
                        "house-prices-advanced-regression-techniques" / "train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" /
                        "gradient_descent")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=500000)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    X_frame, y, audit = prepare_training_features(frame)
    X = X_frame.to_numpy()
    means, scales = X.mean(axis=0), X.std(axis=0, ddof=0)
    if (scales == 0).any():
        raise ValueError("A selected feature is constant.")
    Z = (X - means) / scales
    A = np.column_stack([np.ones(len(y)), Z])
    theta, history, optimization = fit_gradient_descent(
        A, y, args.learning_rate, args.tolerance, args.max_iterations)
    predictions = A @ theta
    raw_weights = theta[1:] / scales
    raw_intercept = float(theta[0] - means @ raw_weights)
    raw_predictions = raw_intercept + X @ raw_weights

    # ONLY AFTER gradient descent: independent verification, never a GD update.
    reference, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    reference_predictions = A @ reference
    final_cost, reference_cost = cost(A, y, theta), cost(A, y, reference)
    pred_diff = float(np.max(abs(predictions-reference_predictions)))
    raw_diff = float(np.max(abs(predictions-raw_predictions)))
    gradient_norm = float(np.linalg.norm(A.T @ (predictions-y) / len(y)))
    checks = {
        "design_full_rank": int(rank) == len(FEATURES)+1,
        "gd_predictions_within_0_001_usd_of_lstsq": pred_diff < .001,
        "gd_cost_within_0_001_usd2_of_lstsq": abs(final_cost-reference_cost) < .001,
        "raw_and_scaled_predictions_within_0_000001_usd": raw_diff < 1e-6,
        "gradient_meets_tolerance": gradient_norm <= args.tolerance,
        "recorded_cost_nonincreasing_to_roundoff":
            bool(np.all(np.diff(history["cost_usd2"]) <= 1e-4)),
    }
    if not all(checks.values()):
        raise AssertionError(f"Verification failed: {checks}")
    metrics = {
        "model": "18-feature linear regression fitted by full-batch gradient descent",
        "target": "SalePrice (USD), no log transform",
        "sample_count": len(y), "feature_count": len(FEATURES),
        "cost_definition": "J = sum((prediction - SalePrice)^2) / (2*m)",
        "training_cost_usd2": final_cost, "training_mse_usd2": 2*final_cost,
        "training_rmse_usd": float(np.sqrt(2*final_cost)),
        "training_r_squared":
            float(1-np.sum((y-predictions)**2)/np.sum((y-y.mean())**2)),
        "initial_cost_usd2": float(history.iloc[0]["cost_usd2"]),
        "mean_prediction_baseline_cost_usd2": float(np.mean((y-y.mean())**2)/2),
        "raw_intercept": raw_intercept, "standardized_intercept": float(theta[0]),
        "optimization": optimization,
        "verification": {
            "checks": checks, "design_rank_including_intercept": int(rank),
            "least_squares_reference_cost_usd2": reference_cost,
            "gd_minus_reference_cost_usd2": final_cost-reference_cost,
            "max_prediction_difference_vs_lstsq_usd": pred_diff,
            "max_standardized_coefficient_difference_vs_lstsq":
                float(np.max(abs(theta-reference))),
            "max_prediction_difference_after_unscaling_usd": raw_diff,
            "direct_gradient_l2_norm": gradient_norm,
            "gradient_based_cost_gap_bound_usd2":
                gradient_norm**2/(2*optimization["smallest_hessian_eigenvalue"]),
        },
        "evaluation_scope": "In-sample fit on all labeled rows; no CV or test evaluation.",
        "source_file": args.data.name,
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    model = {
        "feature_order": FEATURES, "target": "SalePrice", "target_units": "USD",
        "target_transformation": "none", "preprocessing_fit_scope": "all training rows",
        "feature_means": means.tolist(), "feature_scales_ddof0": scales.tolist(),
        "standardized_intercept": float(theta[0]),
        "standardized_weights": theta[1:].tolist(),
        "raw_intercept": raw_intercept, "raw_weights": raw_weights.tolist(),
        "preprocessing": audit,
    }
    coefficients = pd.DataFrame({
        "parameter": ["b"]+[f"w{j}" for j in range(1,len(FEATURES)+1)],
        "feature": ["Intercept"]+FEATURES,
        "raw_input_coefficient": [raw_intercept]+raw_weights.tolist(),
        "standardized_input_coefficient": theta.tolist(),
    })
    scaling = pd.DataFrame({"feature": FEATURES, "training_mean": means,
                            "training_std_ddof0": scales})
    fitted = pd.DataFrame({"Id": frame["Id"], "SalePrice": y,
                          "FittedSalePrice": predictions,
                          "Residual_actual_minus_fitted": y-predictions})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in [("coefficients.csv", coefficients),
                        ("feature_scaling.csv", scaling),
                        ("convergence.csv", history),
                        ("training_predictions.csv", fitted)]:
        table.to_csv(args.output/name, index=False, float_format="%.15g")
    write_json(args.output/"metrics.json", metrics)
    write_json(args.output/"model.json", model)
    write_report(args.output, metrics, coefficients, raw_weights, audit)
    print(json.dumps({"cost": final_cost, "training_rmse": metrics["training_rmse_usd"],
                      "iterations": optimization["iterations"],
                      "alpha": optimization["learning_rate"],
                      "checks": checks}, indent=2))

if __name__ == "__main__":
    main()
