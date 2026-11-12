# Gradient-descent fit: current 18-feature model

**Training complete.** Full-batch gradient descent reached the minimum training
cost to numerical precision for the current 18-feature model, using all
1,460 labeled houses. The target is SalePrice in US dollars.

## Results

| Quantity | Value |
|---|---:|
| Training cost J = MSE / 2 (USD squared) | 609,081,328.154381 |
| Training RMSE (USD) | 34,902.186985 |
| Training R-squared | 0.806849 |
| Initial cost (all coefficients zero) | 19,519,633,853.882877 |
| Mean-price baseline cost | 3,153,394,292.674650 |
| Learning rate | 0.180802912458 |
| Iterations | 134,062 |

These are **training results**, not out-of-sample accuracy. Five-fold comparison
of the basic and 18-feature models remains to be done. NearISU has not been
constructed and is not included in this fit.

## Coefficients for the original feature units

Apply the coding and construction in [outline.md](../../outline.md), multiply
each unscaled feature by its matching coefficient, and add the intercept:

$$
\hat y_i = b + w_1\mathrm{LotArea}_i + w_2\mathrm{OverallQual}_i
+ \cdots + w_{18}\mathrm{RenovationFreshness}_i.
$$

| Parameter | Feature | Coefficient (original feature units) |
|---|---|---:|
| b | Intercept | -2,475,293.010243 |
| w1 | LotArea | 0.506502 |
| w2 | OverallQual | 1,746.340744 |
| w3 | OverallCond | 5,080.353399 |
| w4 | YearBuilt | 1,217.720558 |
| w5 | YearRemodAdd | 29.027819 |
| w6 | ExterCond | -2,087.973246 |
| w7 | BsmtCond | 403.910865 |
| w8 | BsmtExposure | 7,376.767408 |
| w9 | HeatingQC | 2,985.796929 |
| w10 | GrLivArea | -22.101324 |
| w11 | Fireplaces | 11,236.875069 |
| w12 | GarageArea | 49.717019 |
| w13 | GarageQual | -2,851.775069 |
| w14 | PavedDrive | 1,611.045700 |
| w15 | PoolQC | -14,884.935521 |
| w16 | Age | 847.284659 |
| w17 | QualityAdjustedLivingArea | 11.059025 |
| w18 | RenovationFreshness | 18,936.473190 |

Values above are rounded. [model.json](model.json) stores full precision;
[coefficients.csv](coefficients.csv) gives both raw-input and standardized-input
coefficients. Both sets produce the same dollar predictions. Do not mix the scales.

A negative coefficient is not automatically an error or a causal finding.
The model includes GrLivArea and GrLivArea × OverallQual, so its slope with
respect to living area, holding other underlying attributes fixed, is
**-22.101324 + 11.059025 × OverallQual**.
Age and YearBuilt are highly correlated; their separate coefficients also need
careful interpretation. No positive-sign restrictions were imposed.

## Preparation and fitting

All rows and all 18 selected features were kept. SalePrice was not logged.
Only a documented absent facility receives score 0. For an unknown rating, the
mode is learned from observed training categories.

| Field | Confirmed absence: score 0 | Unknown entries imputed | Training mode |
|---|---:|---:|---|
| ExterCond | 0 | 0 | TA |
| BsmtCond | 37 | 0 | TA |
| BsmtExposure | 37 | 1 | No |
| HeatingQC | 0 | 0 | Ex |
| GarageQual | 81 | 0 | TA |
| PavedDrive | 0 | 0 | Y |
| PoolQC | 1453 | 0 | Gd |

Negative ages were capped at zero: 0 construction-age
records and 1 remodeling-age records.
The three constructed features follow outline.md exactly.

Each feature was standardized: z_ij = (x_ij - mean_j) / std_j, using ddof=0.
Imputation and scaling were fitted on this training sample only. In future
cross-validation, they must be learned again within each training fold.

Starting from b = 0 and all weights = 0, we used simultaneous updates:

$$
J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m
\frac{1}{2}(\vec w\cdot\vec z_i+b-y_i)^2.
$$

$$
\frac{\partial J}{\partial w_j}=\frac{1}{m}
\sum_{i=1}^m(\hat y_i-y_i)z_{ij},
\qquad
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_{i=1}^m(\hat y_i-y_i).
$$

$$
w_j^{new}=w_j^{old}-\alpha\frac{\partial J}{\partial w_j},
\qquad b^{new}=b^{old}-\alpha\frac{\partial J}{\partial b}.
$$

We go through w1 to w18, calculate each derivative at the old coefficients, and
then update all weights and b together. Fixed matrix products in this same
gradient are cached to make the iterations faster.

The learning rate is 1/L, where L is the largest eigenvalue of this quadratic
cost's Hessian after standardization. We stop when the full gradient norm is at
most 1e-08; small cost changes alone do not stop training.
There is no regularization.

Convert back to original units using w_raw,j = w_scaled,j / std_j and
b_raw = b_scaled - sum_j(w_raw,j × mean_j). Scaling changes the units used during
optimization, not the set of possible fitted linear models.

## Verification

- The design including the intercept has full rank
  19; this quadratic cost has a unique
  minimizer for the fixed data and features.
- An independent least-squares calculation was run **after** GD, only to verify
  the result. Its largest fitted-price difference from GD is
  **0.0000007527 USD**.
- Raw-input and standardized-input predictions differ by at most
  **0.0000000011 USD**.
- The direct residual-based gradient norm is
  **9.999e-09**.
- Recorded cost is nonincreasing up to floating-point roundoff.
- These checks establish optimization accuracy, not out-of-sample performance.

## Reproduce

From the repository root (Python 3.12 was used):

```bash
python -m pip install -r requirements.txt
python scripts/fit_gradient_descent.py
```

[Source code](../../scripts/fit_gradient_descent.py) implements preprocessing,
GD, checks and output generation.
[metrics.json](metrics.json) records settings, checks, software versions and the
input-file hash. [feature_scaling.csv](feature_scaling.csv) records means and
standard deviations. [convergence.csv](convergence.csv) records iterations 0-10,
every 500, and the last. [training_predictions.csv](training_predictions.csv)
contains fitted training prices and residuals, not competition-test predictions.
