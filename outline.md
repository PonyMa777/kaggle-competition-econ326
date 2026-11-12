# Predicting House Prices in Ames, Iowa — Project Outline

## Track

Advanced track (prediction-focused).

## Research Statement

Homebuyers and real estate professionals need reliable estimates of house prices when comparing properties with different features. This project asks: How accurately can a linear regression model predict sale prices in Ames, Iowa, using house size, quality, and age etc? Does adding renovation and other property features improve out-of-sample accuracy? We will use gradient descent to improve our model and measure accuracy by log RMSE; lower values indicate better predictions.

## Data

We use the Ames Housing dataset (De Cock, 2011), distributed through Kaggle's *House Prices: Advanced Regression Techniques* competition. It contains residential sales in Ames, Iowa, from 2006 to 2010 and 79 explanatory variables. The labeled training file has 1,460 houses; the competition test file has 1,459 houses without observed sale prices. We consider 19 selected raw features and three constructed features. Several selected categorical fields have missing entries, often because a house has no basement, fireplace, garage, or pool. We distinguish a documented absence from an unknown value; any remaining missing values are handled using rules fitted on the training portion of each cross-validation fold. A few records have a construction or remodeling year later than the sale year; we cap the corresponding negative age at zero when constructing age-based features and note this data anomaly. We compare candidate models using five-fold cross-validation on the labeled data, then predict the competition test set.

## Variables

| Variable | Construction | 中文释义与编码 | Reason for inclusion |
|---|---|---|---|
| log(SalePrice) | Log of sale price in USD; outcome, not a feature. | 成交价的自然对数；预测目标 | Prices are skewed, and the Kaggle evaluation metric uses log prices. |
| LotArea | Lot size in square feet. | 土地面积，平方英尺 | More land may increase value beyond the house itself. |
| OverallQual | Overall material and finish rating, 1–10. | 整体材料与装修质量，1–10 分 | Captures broad construction quality. |
| OverallCond | Overall condition rating, 1–10. | 房屋整体当前状况，1–10 分 | Captures current condition separately from quality. |
| YearBuilt | Original construction year. | 最初建造年份 | Captures differences between older and newer homes. |
| YearRemodAdd | Most recent remodeling year. | 最近改造年份；未改造时等于建造年份 | Captures how recently the home was updated. |
| ExterQual | Exterior quality rating. | 外部材料质量；Po=1、Fa=2、TA=3、Gd=4、Ex=5 | Buyers may value better exterior materials. |
| ExterCond | Exterior condition rating. | 外部材料当前状况；Po=1、Fa=2、TA=3、Gd=4、Ex=5 | Captures exterior wear and maintenance needs. |
| BsmtQual | Basement quality rating; 0 if no basement. | 地下室高度对应的质量；Po=1 至 Ex=5；无地下室=0 | Distinguishes usable, high-quality basements. |
| BsmtCond | Basement condition rating; 0 if no basement. | 地下室当前状况；Po=1 至 Ex=5；无地下室=0 | Captures basement wear or damage. |
| BsmtExposure | Basement exposure rating; 0 if no basement. | 地下室采光程度；No=1、Mn=2、Av=3、Gd=4；无地下室=0 | Better exposure may improve light and usability. |
| HeatingQC | Heating quality and condition rating. | 供暖质量与状况；Po=1 至 Ex=5 | Captures comfort and expected repair costs. |
| GrLivArea | Above-ground living area in square feet. | 地上居住面积，平方英尺 | Measures the main usable living space. |
| Fireplaces | Number of fireplaces. | 壁炉数量 | Measures the presence and extent of this amenity. |
| FireplaceQu | Fireplace quality rating; 0 if no fireplace. | 壁炉质量；Po=1 至 Ex=5；无壁炉=0 | Distinguishes the value of existing fireplaces. |
| GarageArea | Garage area in square feet; 0 if no garage. | 车库面积，平方英尺；无车库=0 | Captures parking and storage space. |
| GarageQual | Garage quality rating; 0 if no garage. | 车库质量；Po=1 至 Ex=5；无车库=0 | Captures quality beyond garage size. |
| GarageCond | Garage condition rating; 0 if no garage. | 车库当前状况；Po=1 至 Ex=5；无车库=0 | Captures maintenance needs beyond garage quality. |
| PavedDrive | Driveway surface category. | 车道铺装；N=1（泥土／碎石）、P=2（部分铺装）、Y=3（已铺装） | May affect convenience and reflect property upkeep. |
| PoolQC | Pool quality rating; 0 if no pool. | 泳池质量；Fa=2、TA=3、Gd=4、Ex=5；无泳池=0 | Tests whether a rare amenity adds predictive information. |
| Age | `max(0, YrSold - YearBuilt)`. | 出售时房龄＝出售年份－建造年份；负值截为0 | Measures the home's age at sale. |
| QualityAdjustedLivingArea | `GrLivArea × OverallQual`. | 质量调整后的地上居住面积＝GrLivArea×OverallQual | Allows the value of additional living area to vary with quality. |
| RenovationFreshness | `1 / (1 + max(0, YrSold - YearRemodAdd))`. | 改造新近程度＝1／(1＋距上次改造年数)；越大越新 | Higher values indicate a more recent renovation at sale. |

**Coding assumptions.** We encode ordered quality and condition levels as numeric scores: `Po=1`, `Fa=2`, `TA=3`, `Gd=4`, and `Ex=5`. Where `NA` documents the absence of a basement, fireplace, garage, or pool, we use 0; a genuinely unknown value is not coded as 0. This assumes equal modeled differences between adjacent levels and places absence below the lowest-quality existing facility. `BsmtExposure` uses `No=1`, `Mn=2`, `Av=3`, `Gd=4`, and 0 for no basement. `PavedDrive` is scored as `N=1` (dirt/gravel), `P=2` (partial pavement), and `Y=3` (paved), assuming equal modeled differences between adjacent levels. `Neighborhood` is only a possible future candidate; it is not yet part of the selected feature list.

## Summary Statistics (training data, N = 1,460)

The table uses the numeric scores defined above for ordered quality and condition fields. Standard deviations are sample standard deviations. The single unknown `BsmtExposure` in the training data is filled with the training mode (`No=1`) for this descriptive table; in cross-validation, imputation rules are fitted separately within each training fold. `PavedDrive` is summarized using the proposed `N=1`, `P=2`, `Y=3` score.

| No. | Variable | Mean | SD | Min | Max |
|---:|---|---:|---:|---:|---:|
| 1 | SalePrice (USD) | 180,921 | 79,443 | 34,900 | 755,000 |
| 2 | log(SalePrice) | 12.024 | 0.400 | 10.460 | 13.535 |
| 3 | LotArea (sq ft) | 10,517 | 9,981 | 1,300 | 215,245 |
| 4 | OverallQual | 6.099 | 1.383 | 1 | 10 |
| 5 | OverallCond | 5.575 | 1.113 | 1 | 9 |
| 6 | YearBuilt | 1971.268 | 30.203 | 1872 | 2010 |
| 7 | YearRemodAdd | 1984.866 | 20.645 | 1950 | 2010 |
| 8 | ExterQual (score) | 3.396 | 0.574 | 2 | 5 |
| 9 | ExterCond (score) | 3.084 | 0.351 | 1 | 5 |
| 10 | BsmtQual (score) | 3.489 | 0.876 | 0 | 5 |
| 11 | BsmtCond (score) | 2.935 | 0.552 | 0 | 4 |
| 12 | BsmtExposure (score) | 1.631 | 1.067 | 0 | 4 |
| 13 | HeatingQC (score) | 4.145 | 0.960 | 1 | 5 |
| 14 | GrLivArea (sq ft) | 1,515 | 525 | 334 | 5,642 |
| 15 | Fireplaces (count) | 0.613 | 0.645 | 0 | 3 |
| 16 | FireplaceQu (score) | 1.825 | 1.811 | 0 | 5 |
| 17 | GarageArea (sq ft) | 472.980 | 213.805 | 0 | 1,418 |
| 18 | GarageQual (score) | 2.810 | 0.723 | 0 | 5 |
| 19 | GarageCond (score) | 2.809 | 0.720 | 0 | 5 |
| 20 | PavedDrive (score) | 2.856 | 0.497 | 1 | 3 |
| 21 | PoolQC (score) | 0.018 | 0.269 | 0 | 5 |
| 22 | Age (years) | 36.548 | 30.250 | 0 | 136 |
| 23 | QualityAdjustedLivingArea | 9,673.956 | 5,186.745 | 334 | 56,420 |
| 24 | RenovationFreshness | 0.185 | 0.279 | 0.016 | 1 |

## Proposed Regression Model

We use linear regression to predict **log sale price**. Let `y_i = log(SalePrice_i)` and let `x_i = (x_{i1}, ..., x_{ip})` contain the preprocessed features selected for a candidate model from the Variables table. The full candidate has up to 22 features (19 raw and three constructed); smaller candidates use subsets. We compare feature sets by five-fold cross-validation.

**1. Prediction function (model).** The full 22-feature candidate is:

$$
\begin{aligned}
\hat y_i=f_{\mathbf w,b}(\mathbf x_i) &= b + w_{1}\,\mathrm{LotArea}_i + w_{2}\,\mathrm{OverallQual}_i + w_{3}\,\mathrm{OverallCond}_i \\
&\quad + w_{4}\,\mathrm{YearBuilt}_i + w_{5}\,\mathrm{YearRemodAdd}_i + w_{6}\,\mathrm{ExterQual}_i \\
&\quad + w_{7}\,\mathrm{ExterCond}_i + w_{8}\,\mathrm{BsmtQual}_i + w_{9}\,\mathrm{BsmtCond}_i \\
&\quad + w_{10}\,\mathrm{BsmtExposure}_i + w_{11}\,\mathrm{HeatingQC}_i + w_{12}\,\mathrm{GrLivArea}_i \\
&\quad + w_{13}\,\mathrm{Fireplaces}_i + w_{14}\,\mathrm{FireplaceQu}_i + w_{15}\,\mathrm{GarageArea}_i \\
&\quad + w_{16}\,\mathrm{GarageQual}_i + w_{17}\,\mathrm{GarageCond}_i + w_{18}\,\mathrm{PavedDrive}_i \\
&\quad + w_{19}\,\mathrm{PoolQC}_i + w_{20}\,\mathrm{Age}_i + w_{21}\,\mathrm{QualityAdjustedLivingArea}_i \\
&\quad + w_{22}\,\mathrm{RenovationFreshness}_i
\end{aligned}
$$

Each `w_j` is the coefficient of the named feature; `b` is the intercept. The quality and condition terms use the numeric scores defined above. Smaller candidate models omit selected terms.

**2. Cost function (training objective).** For the `m` labeled houses in the current training portion,

$$
J(\mathbf w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y_i-y_i)^2.
$$

We minimize squared errors in **log prices**, consistent with the log RMSE used for validation. The factor `1/2` simplifies the derivative without changing the minimizing coefficients.

**3. Gradient-descent updates.** At each iteration, with learning rate `\alpha`,

$$
w_j^{\mathrm{new}}=w_j^{\mathrm{old}}-\alpha\frac{\partial J}{\partial w_j},\qquad
b^{\mathrm{new}}=b^{\mathrm{old}}-\alpha\frac{\partial J}{\partial b},
$$

where

$$
\frac{\partial J}{\partial w_j}=\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i)x_{ij},\qquad
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i).
$$

If we scale features for gradient descent, we fit the scaling parameters on each cross-validation training fold and apply them to its validation fold. Gradient descent estimates the coefficients for a given feature set; cross-validation compares feature sets using log RMSE. After selecting a model, we fit it on all labeled houses and exponentiate predicted log prices to obtain dollar-price predictions for the competition test set.
## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
- Kaggle. *House Prices: Advanced Regression Techniques*. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Notes on Missing Values and Categorical Features / 缺失值与定性变量说明

1. **Unknown basement exposure / 地下室采光未知。** One training house has a basement, but its `BsmtExposure` value is missing. This is an unknown rating, not “no basement.” We will fit an imputation rule using only the training portion of each cross-validation fold and apply it to that fold's validation portion. For the final model, we will fit the rule on the full training set and apply it to the competition test set. / 训练集中有一套房屋存在地下室，但 `BsmtExposure` 未记录；这表示采光等级未知，不表示没有地下室。每折交叉验证只用训练部分确定填补规则，再用于该折的验证部分；最终用完整训练集确定规则，再用于竞赛测试集。

2. **Categorical features / 定性特征。** For ordered quality and condition variables, we propose numeric scores (Po=1, Fa=2, TA=3, Gd=4, Ex=5; 0 only for a documented absence) to keep the initial model manageable. This assumes equal modeled differences between adjacent levels and that absence is below the lowest existing level. `PavedDrive` is also scored numerically (`N=1`, `P=2`, `Y=3`), assuming equal modeled differences between adjacent paving levels. The summary-statistics table reports these proposed numeric scores. If we use feature scaling, we will fit its parameters on each fold's training portion only. / 对有顺序的质量和状况变量，我们拟采用数值编码（Po=1、Fa=2、TA=3、Gd=4、Ex=5；仅在确认设施不存在时记为0），使初始模型更简洁。这假设相邻等级在模型中的差距相等，且没有设施低于质量最差但仍存在的设施。`PavedDrive` 也按 `N=1`、`P=2`、`Y=3` 编码，假设相邻铺装等级在模型中的差距相等。描述性统计表报告这些拟采用的数值评分。如使用特征缩放，也只用每折的训练部分计算缩放参数。
## My Open Questions / 我的待解决问题

1. **Gradient descent, log prices, and log RMSE.** If I use gradient descent to fit the regression, should the outcome still be `log(SalePrice)` and the validation metric still be log RMSE? Does the optimization method change either choice? / 如果我用梯度下降拟合回归，是否仍应预测 `log(SalePrice)`、并用 log RMSE 评估？优化方法会改变这两个选择吗？
2. **Feature types.** After applying the proposed ordinal encodings, are all 22 current candidate features numeric inputs to the model? / 按目前的等级编码处理后，现有 22 个候选特征是否都能作为模型的数值输入？
3. **Learning rate and scaling.** How should I choose the learning rate and decide which features to scale for gradient descent, without using validation data to fit the scaling rules? / 接下来怎样选择学习率、决定哪些特征需要缩放，并确保缩放规则只由训练数据确定？
4. **Neighborhood comparison.** I still need a comparison between a model without `Neighborhood` and one with it. Should I represent its multiple categories with neighborhood dummies, or define a single near-campus indicator, and does adding location improve cross-validated log RMSE? / 我还需要比较“不加入 `Neighborhood`”与“加入社区位置”的模型。应该为多个社区分别建 dummy，还是定义一个“靠近大学”的 0/1 变量？加入位置后，交叉验证的 log RMSE 是否改善？
