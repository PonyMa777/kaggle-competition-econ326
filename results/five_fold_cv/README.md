# Five-fold cross-validation: basic versus full model

**B_full has the lowest mean validation RMSE among the two compared models.**
The 18-feature model reduces mean RMSE by USD 3,836.09
(9.42%) relative to the basic model.

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
| B_full | 18 | 36,865.74 | 9,232.28 | 37,779.24 |
| A_basic | 3 | 40,701.83 | 4,516.15 | 40,901.77 |
| Mean_price_baseline | 0 | 79,273.77 | 5,854.54 | 79,446.53 |

Selection uses the **arithmetic mean of the five RMSEs**. SD is the sample
standard deviation across folds, not a confidence interval. Pooled OOF RMSE
combines all held-out squared errors before taking the square root; it differs
from mean fold RMSE. Neither is a training error.

| Fold | Mean-price baseline | A: 3 features | B: 18 features |
|---|---:|---:|---:|
| 1 | 71,259.03 | 35,749.44 | 29,521.74 |
| 2 | 79,395.64 | 38,273.94 | 29,949.63 |
| 3 | 76,308.65 | 43,767.54 | 48,819.56 |
| 4 | 83,104.00 | 38,830.75 | 31,177.10 |
| 5 | 86,301.51 | 46,887.46 | 44,860.66 |

## Procedure

1. Shuffle the labeled rows once with seed 42 and divide them into
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

B has lower RMSE in 4 of the five folds. Folds where B has higher RMSE: [3].
Read the fold standard deviations alongside the mean errors.

This identifies the better of **these two feature sets on this five-fold split**.
It does not prove that every added feature helps, or find the best of all possible
feature combinations. The CV errors also guide model selection; they are not an
independent final test evaluation. The unlabeled competition test set was not used.

Fold coefficients belong to different training subsets; do not average them to
form the final model. Fit the chosen feature set on all labeled rows.
The existing [full-data 18-feature fit](../gradient_descent/README.md) is stored
separately with its training error clearly labeled.

## Reproduce and inspect

```bash
python -m pip install -r requirements.txt
python scripts/cross_validate_models.py
```

- [Source](../../scripts/cross_validate_models.py)
- [Comparison](comparison.csv) and [per-fold metrics](fold_metrics.csv)
- [Exact fold assignments](fold_assignments.csv)
- [Out-of-fold predictions](out_of_fold_predictions.csv)
- [Fold preprocessing parameters and coefficients](fold_models.json)
- [Seed, input hash, and verification](metadata.json)
