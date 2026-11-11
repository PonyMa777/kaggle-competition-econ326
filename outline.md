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

| Variable | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| SalePrice (USD) | 180,921 | 79,443 | 34,900 | 755,000 |
| log(SalePrice) | 12.024 | 0.400 | 10.460 | 13.535 |
| LotArea (sq ft) | 10,517 | 9,981 | 1,300 | 215,245 |
| OverallQual | 6.099 | 1.383 | 1 | 10 |
| OverallCond | 5.575 | 1.113 | 1 | 9 |
| YearBuilt | 1971.268 | 30.203 | 1872 | 2010 |
| YearRemodAdd | 1984.866 | 20.645 | 1950 | 2010 |
| ExterQual (score) | 3.396 | 0.574 | 2 | 5 |
| ExterCond (score) | 3.084 | 0.351 | 1 | 5 |
| BsmtQual (score) | 3.489 | 0.876 | 0 | 5 |
| BsmtCond (score) | 2.935 | 0.552 | 0 | 4 |
| BsmtExposure (score) | 1.631 | 1.067 | 0 | 4 |
| HeatingQC (score) | 4.145 | 0.960 | 1 | 5 |
| GrLivArea (sq ft) | 1,515 | 525 | 334 | 5,642 |
| Fireplaces (count) | 0.613 | 0.645 | 0 | 3 |
| FireplaceQu (score) | 1.825 | 1.811 | 0 | 5 |
| GarageArea (sq ft) | 472.980 | 213.805 | 0 | 1,418 |
| GarageQual (score) | 2.810 | 0.723 | 0 | 5 |
| GarageCond (score) | 2.809 | 0.720 | 0 | 5 |
| PavedDrive (score) | 2.856 | 0.497 | 1 | 3 |
| PoolQC (score) | 0.018 | 0.269 | 0 | 5 |
| Age (years) | 36.548 | 30.250 | 0 | 136 |
| QualityAdjustedLivingArea | 9,673.956 | 5,186.745 | 334 | 56,420 |
| RenovationFreshness | 0.185 | 0.279 | 0.016 | 1 |

## Proposed Regression Model

Full specification (Model C):

$$
\log(\text{SalePrice}_i) = \beta_0 + \beta_1 \log(\text{GrLivArea}_i) + \beta_2\,\text{OverallQual}_i + \beta_3\,\text{Age}_i + \beta_4\,\text{TotalBsmtSF}_i + \beta_5\,\text{GarageCars}_i + \beta_6\,\text{TotalBath}_i + \beta_7 \log(\text{LotArea}_i) + \beta_8\,\text{KitEx}_i + \beta_9\,\text{KitGd}_i + \beta_{10}\,\text{KitFa}_i + \beta_{11}\,\text{YearsSinceRemod}_i + u_i
$$

Model A keeps only β₁–β₃; Model B adds β₄–β₆; Model C adds the rest. We estimate each model by OLS, which minimizes the mean squared error in log price, and compare the three models on log RMSE using the same five-fold cross-validation split of the training data. We choose the simplest model whose mean CV error is within one standard error of the best. The log–log form gives β₁ and β₇ an elasticity interpretation, and the other coefficients are approximate percentage effects. We then refit the chosen model on all 1,460 houses, predict the test set, and convert predictions back to dollars with exp(·).

## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
- Kaggle. *House Prices: Advanced Regression Techniques*. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Notes on Missing Values and Categorical Features / 缺失值与定性变量说明

1. **Unknown basement exposure / 地下室采光未知。** One training house has a basement, but its `BsmtExposure` value is missing. This is an unknown rating, not “no basement.” We will fit an imputation rule using only the training portion of each cross-validation fold and apply it to that fold's validation portion. For the final model, we will fit the rule on the full training set and apply it to the competition test set. / 训练集中有一套房屋存在地下室，但 `BsmtExposure` 未记录；这表示采光等级未知，不表示没有地下室。每折交叉验证只用训练部分确定填补规则，再用于该折的验证部分；最终用完整训练集确定规则，再用于竞赛测试集。

2. **Categorical features / 定性特征。** For ordered quality and condition variables, we propose numeric scores (Po=1, Fa=2, TA=3, Gd=4, Ex=5; 0 only for a documented absence) to keep the initial model manageable. This assumes equal modeled differences between adjacent levels and that absence is below the lowest existing level. `PavedDrive` is also scored numerically (`N=1`, `P=2`, `Y=3`), assuming equal modeled differences between adjacent paving levels. The summary-statistics table reports these proposed numeric scores. If we use feature scaling, we will fit its parameters on each fold's training portion only. / 对有顺序的质量和状况变量，我们拟采用数值编码（Po=1、Fa=2、TA=3、Gd=4、Ex=5；仅在确认设施不存在时记为0），使初始模型更简洁。这假设相邻等级在模型中的差距相等，且没有设施低于质量最差但仍存在的设施。`PavedDrive` 也按 `N=1`、`P=2`、`Y=3` 编码，假设相邻铺装等级在模型中的差距相等。描述性统计表报告这些拟采用的数值评分。如使用特征缩放，也只用每折的训练部分计算缩放参数。