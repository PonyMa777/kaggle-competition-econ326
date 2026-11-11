# Predicting House Prices in Ames, Iowa — Project Outline

## Track

Advanced track (prediction-focused).

## Research Statement

How accurately can a linear regression predict a house's sale price in Ames, Iowa, from its size, quality, and age, and how much does adding basement, garage, bathroom, lot, kitchen, and renovation information improve out-of-sample accuracy? We measure accuracy with the root-mean-squared error between the logs of predicted and actual sale prices (log RMSE), the metric used by the Kaggle competition; lower is better.

## Data

We use the Ames Housing dataset (De Cock, 2011), distributed through the Kaggle competition *House Prices: Advanced Regression Techniques*. It records residential sales in Ames, Iowa, from 2006 to 2010 and has 79 explanatory variables. The training file has 1,460 houses with observed sale prices; the test file has 1,459 houses without prices, for which we submit predictions. In the training data, none of our chosen variables has missing values. In the test data, five cells are missing: we fill basement values with 0 where the house has no basement, and use the training median or mode otherwise.

## Variables

| Variable | Construction | Justification |
|---|---|---|
| log(SalePrice) | log of sale price (USD) | Outcome. Prices are right-skewed, and the evaluation metric is defined in logs. |
| GrLivArea | above-ground living area (sq ft), entered as a log | Size is the main driver of price; in logs, its coefficient is an elasticity. |
| OverallQual | overall material and finish rating, 1–10 | Summarizes construction quality, which buyers pay for beyond size. |
| Age | YrSold − YearBuilt | Older houses tend to sell for less because of depreciation and outdated design. |
| TotalBsmtSF | basement area (sq ft); 0 = no basement | Basement space is usable area not counted in GrLivArea. |
| GarageCars | garage capacity in cars; 0 = no garage | Captures garage size, a valued amenity in Iowa. |
| TotalBath | FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath | Combines four bathroom counts into one measure of functionality. |
| LotArea | lot size (sq ft), entered as a log | Land has value; logs reduce the influence of a few very large lots. |
| KitchenQual | Ex / Gd / TA / Fa, as dummies (TA omitted) | Kitchen quality is a key renovation feature that OverallQual may not fully capture. |
| YearsSinceRemod | YrSold − YearRemodAdd (equals Age if never remodeled) | Recent renovation should raise price conditional on age. |

## Summary Statistics (training data, N = 1,460)

| Variable | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| SalePrice (USD) | 180,921 | 79,443 | 34,900 | 755,000 |
| log(SalePrice) | 12.02 | 0.40 | 10.46 | 13.53 |
| GrLivArea (sq ft) | 1,515 | 525 | 334 | 5,642 |
| OverallQual | 6.10 | 1.38 | 1 | 10 |
| Age (years) | 36.55 | 30.25 | 0 | 136 |
| TotalBsmtSF (sq ft) | 1,057 | 439 | 0 | 6,110 |
| GarageCars | 1.77 | 0.75 | 0 | 4 |
| TotalBath | 2.21 | 0.79 | 1 | 6 |
| LotArea (sq ft) | 10,517 | 9,981 | 1,300 | 215,245 |
| YearsSinceRemod (years) | 22.95 | 20.64 | 0 | 60 |
| KitchenQual = Ex | 0.07 | 0.25 | 0 | 1 |
| KitchenQual = Gd | 0.40 | 0.49 | 0 | 1 |
| KitchenQual = TA | 0.50 | 0.50 | 0 | 1 |
| KitchenQual = Fa | 0.03 | 0.16 | 0 | 1 |

## Proposed Regression Model

Full specification (Model C):

$$
\log(\text{SalePrice}_i) = \beta_0 + \beta_1 \log(\text{GrLivArea}_i) + \beta_2\,\text{OverallQual}_i + \beta_3\,\text{Age}_i + \beta_4\,\text{TotalBsmtSF}_i + \beta_5\,\text{GarageCars}_i + \beta_6\,\text{TotalBath}_i + \beta_7 \log(\text{LotArea}_i) + \beta_8\,\text{KitEx}_i + \beta_9\,\text{KitGd}_i + \beta_{10}\,\text{KitFa}_i + \beta_{11}\,\text{YearsSinceRemod}_i + u_i
$$

Model A keeps only β₁–β₃; Model B adds β₄–β₆; Model C adds the rest. We estimate each model by OLS, which minimizes the mean squared error in log price, and compare the three models on log RMSE using the same five-fold cross-validation split of the training data. We choose the simplest model whose mean CV error is within one standard error of the best. The log–log form gives β₁ and β₇ an elasticity interpretation, and the other coefficients are approximate percentage effects. We then refit the chosen model on all 1,460 houses, predict the test set, and convert predictions back to dollars with exp(·).

## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
- Kaggle. *House Prices: Advanced Regression Techniques*. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
