# Predicting House Prices in Ames, Iowa — Project Outline

## Track

Advanced track (prediction-focused).

## Research Statement

Homebuyers and real estate professionals need reliable estimates of house prices when comparing properties with different features. This project asks: How accurately can house size, quality, and age predict sale prices in Ames, Iowa? Does adding information on renovations, amenities, and neighborhood location improve prediction accuracy for out-of-sample houses? We will use ordinary linear regression (OLS) as a benchmark and compare it with a machine-learning regression approach. Both will predict the natural logarithm of sale price. We will use five-fold cross-validation to compare candidate models and select the model with the lowest mean validation log RMSE. This is a prediction study; associations between features and prices are not interpreted as causal effects.

## Data

We use the Ames Housing dataset (De Cock, 2011), distributed through Kaggle's *House Prices: Advanced Regression Techniques* competition. It contains residential sales in Ames, Iowa, from 2006 to 2010 and 79 explanatory variables. The labeled training file has 1,460 houses; the competition test file has 1,459 houses without observed sale prices. We consider 15 selected raw features and three constructed features. Several selected categorical fields have missing entries, often because a house has no basement, garage, or pool. We distinguish a documented absence from an unknown value. Remaining unknown categorical values will be imputed using the mode calculated only from the training portion of each cross-validation fold. (**Note: discuss this imputation approach with the professor during the next office hour.**)

A few records have a construction or remodeling year later than the sale year. When constructing age-based features, we set the resulting negative ages to zero and document this data anomaly.

We compare candidate models using five-fold cross-validation on the labeled data.

## Variables

| No. | Feature | Construction | Description and Coding | Reason for inclusion |
|---:|---|---|---|---|
| 1 | log(SalePrice) | Natural logarithm of the observed sale price in USD; outcome, not a feature. | Prediction target: y = ln(SalePrice). | Makes errors reflect relative rather than absolute price differences. |
| 2 | LotArea | Lot size in square feet. | Lot area in square feet. | More land may increase value beyond the house itself. |
| 3 | OverallQual | Overall material and finish rating, 1–10. | Overall material and finish quality, rated 1–10. | Captures broad construction quality. |
| 4 | OverallCond | Overall condition rating, 1–10. | Overall current condition, rated 1–10. | Captures current condition separately from quality. |
| 5 | YearBuilt | Original construction year. | Original construction year. | Captures differences between older and newer homes. |
| 6 | YearRemodAdd | Most recent remodeling year. | Most recent remodeling year; equals the construction year if never remodeled. | Captures how recently the home was updated. |
| 7 | ExterCond | Exterior condition rating. | Current exterior condition; Po=1, Fa=2, TA=3, Gd=4, Ex=5. | Captures exterior wear and maintenance needs. |
| 8 | BsmtCond | Basement condition rating; 0 if no basement. | Current basement condition; Po=1 through Ex=5; no basement=0. | Captures basement wear or damage. |
| 9 | BsmtExposure | Basement exposure rating; 0 if no basement. | Basement exposure; No=1, Mn=2, Av=3, Gd=4; no basement=0. | Better exposure may improve light and usability. |
| 10 | HeatingQC | Heating quality and condition rating. | Heating quality and condition; Po=1 through Ex=5. | Captures comfort and expected repair costs. |
| 11 | GrLivArea | Above-ground living area in square feet. | Above-ground living area in square feet. | Measures the main usable living space. |
| 12 | Fireplaces | Number of fireplaces. | Number of fireplaces. | Measures the presence and extent of this amenity. |
| 13 | GarageArea | Garage area in square feet; 0 if no garage. | Garage area in square feet; no garage=0. | Captures parking and storage space. |
| 14 | GarageQual | Garage quality rating; 0 if no garage. | Garage quality; Po=1 through Ex=5; no garage=0. | Captures quality beyond garage size. |
| 15 | PavedDrive | Driveway surface category. | Driveway paving; N=1 (dirt/gravel), P=2 (partial pavement), Y=3 (paved). | May affect convenience and reflect property upkeep. |
| 16 | PoolQC | Pool quality rating; 0 if no pool. | Pool quality; Fa=2, TA=3, Gd=4, Ex=5; no pool=0. | Tests whether a rare amenity adds predictive information. |
| 17 | Age | `max(0, YrSold - YearBuilt)`. | Age at sale = sale year minus construction year, with negative values capped at 0. | Measures the home's age at sale. |
| 18 | QualityAdjustedLivingArea | `GrLivArea × OverallQual`. | Quality-adjusted living area = GrLivArea × OverallQual. | Allows the value of additional living area to vary with quality. |
| 19 | RenovationFreshness | `1 / (1 + max(0, YrSold - YearRemodAdd))`. | Renovation freshness = 1 / (1 + years since remodeling); higher values indicate a more recent renovation. | Higher values indicate a more recent renovation at sale. |

**Coding assumptions.** We encode ordered quality and condition levels as numeric scores: `Po=1`, `Fa=2`, `TA=3`, `Gd=4`, and `Ex=5`. Where `NA` documents the absence of a basement, garage, or pool, we use 0; a genuinely unknown value is not coded as 0. This assumes equal modeled differences between adjacent levels and places absence below the lowest-quality existing facility. `BsmtExposure` uses `No=1`, `Mn=2`, `Av=3`, `Gd=4`, and 0 for no basement. `PavedDrive` is scored as `N=1` (dirt/gravel), `P=2` (partial pavement), and `Y=3` (paved), assuming equal modeled differences between adjacent levels. `Neighborhood` is only a possible future candidate; it is not yet part of the selected feature list.

## Summary Statistics (training data, N = 1,460)

The table uses the numeric scores defined above for ordered quality and condition fields. Standard deviations are sample standard deviations. The single unknown `BsmtExposure` in the training data is filled with the training mode (`No=1`) for this descriptive table; in cross-validation, imputation rules are fitted separately within each training fold. `PavedDrive` is summarized using the proposed `N=1`, `P=2`, `Y=3` score.

| No. | Variable | Mean | SD | Min | Max |
|---:|---|---:|---:|---:|---:|
| 1 | SalePrice (USD) | 180,921 | 79,443 | 34,900 | 755,000 |
| 2 | log(SalePrice) | 12.024 | 0.399 | 10.460 | 13.534 |
| 3 | LotArea (sq ft) | 10,517 | 9,981 | 1,300 | 215,245 |
| 4 | OverallQual | 6.099 | 1.383 | 1 | 10 |
| 5 | OverallCond | 5.575 | 1.113 | 1 | 9 |
| 6 | YearBuilt | 1971.268 | 30.203 | 1872 | 2010 |
| 7 | YearRemodAdd | 1984.866 | 20.645 | 1950 | 2010 |
| 8 | ExterCond (score) | 3.084 | 0.351 | 1 | 5 |
| 9 | BsmtCond (score) | 2.935 | 0.552 | 0 | 4 |
| 10 | BsmtExposure (score) | 1.631 | 1.067 | 0 | 4 |
| 11 | HeatingQC (score) | 4.145 | 0.960 | 1 | 5 |
| 12 | GrLivArea (sq ft) | 1,515 | 525 | 334 | 5,642 |
| 13 | Fireplaces (count) | 0.613 | 0.645 | 0 | 3 |
| 14 | GarageArea (sq ft) | 472.980 | 213.805 | 0 | 1,418 |
| 15 | GarageQual (score) | 2.810 | 0.723 | 0 | 5 |
| 16 | PavedDrive (score) | 2.856 | 0.497 | 1 | 3 |
| 17 | PoolQC (score) | 0.018 | 0.269 | 0 | 5 |
| 18 | Age (years) | 36.548 | 30.250 | 0 | 136 |
| 19 | QualityAdjustedLivingArea | 9,673.956 | 5,186.745 | 334 | 56,420 |
| 20 | RenovationFreshness | 0.185 | 0.279 | 0.016 | 1 |

## Proposed Regression Model

We use linear regression to predict **the natural logarithm of sale price**. Let `y_i = ln(SalePrice_i)` be the observed log price and `ŷ_i` its predicted log price. This assumes log price is linear in the selected features; it does not require taking logs of the features. Let $\vec{x}_i=(x_{i1},\ldots,x_{ip})$ contain the preprocessed features selected for a candidate model from the Variables table. The full candidate has up to 18 features (15 raw and three constructed); smaller candidates use subsets. We compare feature sets by five-fold cross-validation.

**1. Prediction function (model).** The full 18-feature candidate is:

$$
\begin{aligned}
\hat y_i=f_{\vec{w},b}(\vec{x}_i) &= b + w_{1}\,\mathrm{LotArea}_i + w_{2}\,\mathrm{OverallQual}_i + w_{3}\,\mathrm{OverallCond}_i \\
&\quad + w_{4}\,\mathrm{YearBuilt}_i + w_{5}\,\mathrm{YearRemodAdd}_i + w_{6}\,\mathrm{ExterCond}_i \\
&\quad + w_{7}\,\mathrm{BsmtCond}_i + w_{8}\,\mathrm{BsmtExposure}_i + w_{9}\,\mathrm{HeatingQC}_i \\
&\quad + w_{10}\,\mathrm{GrLivArea}_i + w_{11}\,\mathrm{Fireplaces}_i + w_{12}\,\mathrm{GarageArea}_i \\
&\quad + w_{13}\,\mathrm{GarageQual}_i + w_{14}\,\mathrm{PavedDrive}_i + w_{15}\,\mathrm{PoolQC}_i \\
&\quad + w_{16}\,\mathrm{Age}_i + w_{17}\,\mathrm{QualityAdjustedLivingArea}_i + w_{18}\,\mathrm{RenovationFreshness}_i
\end{aligned}
$$

Each `w_j` is the coefficient of the named feature; `b` is the intercept. The quality and condition terms use the numeric scores defined above. Smaller candidate models omit selected terms.

**Vectorized form (equivalent to the expanded function).** For house `i`, arrange the 18 features in the same order as above into a vector, and arrange their coefficients into a matching weight vector:

$$
\vec{x}_i=\left(\mathrm{LotArea}_i,\ \mathrm{OverallQual}_i,\ \ldots,\ \mathrm{RenovationFreshness}_i\right),
$$

$$
\vec{w}=\left(w_1,w_2,\ldots,w_{18}\right).
$$

The prediction function is:

$$
\hat y_i=\vec{w}\cdot\vec{x}_i+b.
$$

The arrows indicate vectors. The dot product multiplies each weight by its corresponding feature and adds the products, so this makes exactly the same prediction as the expanded equation above.

**2. Cost function (training objective).** For the `m` labeled houses in the current training portion,

$$
J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}\left(f_{\vec{w},b}(\vec{x}_i)-\ln(\mathrm{SalePrice}_i)\right)^2.
$$

We minimize squared errors in **log sale prices**. On a held-out fold containing $m_v$ houses, we calculate:

$$
\mathrm{log\ RMSE}=\sqrt{\frac{1}{m_v}\sum_{i\in\mathrm{validation}}\left(\hat y_i-\ln(\mathrm{SalePrice}_i)\right)^2}.
$$

We compare the mean of the five validation scores. Log RMSE is not measured in dollars. The factor `1/2` simplifies the derivative without changing the minimizing coefficients.

**3. Gradient-descent updates.** We will go through each weight, from $w_1$ to $w_n$, and calculate its partial derivative, where $n$ is the number of features (18 in the full model). With learning rate $\alpha$, update each weight and the intercept:

$$
w_i^{\mathrm{new}}=w_i^{\mathrm{old}}-\alpha\frac{\partial J}{\partial w_i},\qquad
b^{\mathrm{new}}=b^{\mathrm{old}}-\alpha\frac{\partial J}{\partial b}.
$$

In this subsection, `i` indexes the weights (`i = 1, ..., n`) and `r` indexes the training houses. The derivatives are:

$$
\frac{\partial J}{\partial w_i}=\frac{1}{m}\sum_{r=1}^{m}\left(f_{\vec{w},b}(\vec{x}_r)-\ln(\mathrm{SalePrice}_r)\right)x_{ri},\qquad
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_{r=1}^{m}\left(f_{\vec{w},b}(\vec{x}_r)-\ln(\mathrm{SalePrice}_r)\right).
$$

Evaluate all derivatives at the old parameter values, then apply the updates simultaneously.

If we scale features for gradient descent, we fit the scaling parameters on each cross-validation training fold and apply them to its validation fold. Gradient descent estimates the coefficients for a given feature set; cross-validation compares candidate models using log RMSE. After selecting a model, we fit it on all labeled houses. To express its log predictions in dollars, use $\widehat{\mathrm{SalePrice}}_i=\exp(\hat y_i)$. The unlabeled competition test set may be used for a separate prediction exercise; the course project evaluates accuracy using held-out folds of the labeled data.

## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
- Kaggle. *House Prices: Advanced Regression Techniques*. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Notes on Missing Values and Categorical Features / 缺失值与定性变量说明

1. **Unknown basement exposure / 地下室采光未知。** One training house has a basement, but its `BsmtExposure` value is missing. This is an unknown rating, not “no basement.” We will fit an imputation rule using only the training portion of each cross-validation fold and apply it to that fold's validation portion. For the final model, we will fit the rule on the full training set and apply it to the competition test set. / 训练集中有一套房屋存在地下室，但 `BsmtExposure` 未记录；这表示采光等级未知，不表示没有地下室。每折交叉验证只用训练部分确定填补规则，再用于该折的验证部分；最终用完整训练集确定规则，再用于竞赛测试集。

2. **Categorical features / 定性特征。** For ordered quality and condition variables, we propose numeric scores (Po=1, Fa=2, TA=3, Gd=4, Ex=5; 0 only for a documented absence) to keep the initial model manageable. This assumes equal modeled differences between adjacent levels and that absence is below the lowest existing level. `PavedDrive` is also scored numerically (`N=1`, `P=2`, `Y=3`), assuming equal modeled differences between adjacent paving levels. The summary-statistics table reports these proposed numeric scores. If we use feature scaling, we will fit its parameters on each fold's training portion only. / 对有顺序的质量和状况变量，我们拟采用数值编码（Po=1、Fa=2、TA=3、Gd=4、Ex=5；仅在确认设施不存在时记为0），使初始模型更简洁。这假设相邻等级在模型中的差距相等，且没有设施低于质量最差但仍存在的设施。`PavedDrive` 也按 `N=1`、`P=2`、`Y=3` 编码，假设相邻铺装等级在模型中的差距相等。描述性统计表报告这些拟采用的数值评分。如使用特征缩放，也只用每折的训练部分计算缩放参数。

## My Open Questions / 我的待解决问题

1. **Feature types.** After applying the proposed ordinal encodings, are all 18 current candidate features numeric inputs to the model? / 按目前的等级编码处理后，现有 18 个候选特征是否都能作为模型的数值输入？
2. **Learning rate and scaling.** How should I choose the learning rate and decide which features to scale for gradient descent, without using validation data to fit the scaling rules? / 接下来怎样选择学习率、决定哪些特征需要缩放，并确保缩放规则只由训练数据确定？
3. **Benchmark comparison / 基准模型比较。** Following the professor's advice, we will use ordinary linear regression estimated by OLS as the benchmark and compare it with a machine-learning regression approach. Does the machine-learning approach achieve a lower mean out-of-sample log RMSE? Both approaches will use the same validation folds, with preprocessing fitted on training rows only. The benchmark specification and the machine-learning method are still to be decided. With identical features and an unregularized squared-error objective, converged gradient descent and OLS produce essentially the same fitted predictions; changing the optimizer alone is not a distinct predictive model. / 根据教授的建议，我们将普通线性回归（OLS）作为基准，与机器学习回归方案比较，检验后者能否降低平均样本外 log RMSE。两者使用相同的验证折，预处理规则只根据训练部分确定。基准模型的具体设定和机器学习方法仍待确定。同样的特征与无正则化平方误差目标下，收敛后的梯度下降与 OLS 的拟合预测基本相同，因此仅更换求解方法不能构成不同的预测模型。
