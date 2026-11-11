# Predicting House Prices in Ames, Iowa — Project Outline

## Track

Advanced track (prediction-focused).

## Research Statement

Homebuyers and real estate professionals need reliable estimates of house prices when comparing properties with different features. This project asks: How accurately can a linear regression model predict sale prices in Ames, Iowa, using house size, quality, and age etc? Does adding renovation and other property features improve out-of-sample accuracy? We will use gradient descent to improve our model and measure accuracy by log RMSE; lower values indicate better predictions.

## Data

We use the Ames Housing dataset (De Cock, 2011), distributed through Kaggle's *House Prices: Advanced Regression Techniques* competition. It contains residential sales in Ames, Iowa, from 2006 to 2010 and 79 explanatory variables. The labeled training file has 1,460 houses; the competition test file has 1,459 houses without observed sale prices. We consider 19 selected raw features and three constructed features. Several selected categorical fields have missing entries, often because a house has no basement, fireplace, garage, or pool. We distinguish a documented absence from an unknown value; any remaining missing values are handled using rules fitted on the training portion of each cross-validation fold. A few records have a construction or remodeling year later than the sale year; we cap the corresponding negative age at zero when constructing age-based features and note this data anomaly. We compare candidate models using five-fold cross-validation on the labeled data, then predict the competition test set.

## Variables

| Variable | Construction | Reason for inclusion |
|---|---|---|
| log(SalePrice) | Log of sale price in USD; outcome, not a feature. | Prices are skewed, and the Kaggle evaluation metric uses log prices. |
| LotArea | Lot size in square feet. | More land may increase value beyond the house itself. |
| OverallQual | Overall material and finish rating, 1–10. | Captures broad construction quality. |
| OverallCond | Overall condition rating, 1–10. | Captures current condition separately from quality. |
| YearBuilt | Original construction year. | Captures differences between older and newer homes. |
| YearRemodAdd | Most recent remodeling year. | Captures how recently the home was updated. |
| ExterQual | Exterior quality rating. | Buyers may value better exterior materials. |
| ExterCond | Exterior condition rating. | Captures exterior wear and maintenance needs. |
| BsmtQual | Basement quality rating; 0 if no basement. | Distinguishes usable, high-quality basements. |
| BsmtCond | Basement condition rating; 0 if no basement. | Captures basement wear or damage. |
| BsmtExposure | Basement exposure rating; 0 if no basement. | Better exposure may improve light and usability. |
| HeatingQC | Heating quality and condition rating. | Captures comfort and expected repair costs. |
| GrLivArea | Above-ground living area in square feet. | Measures the main usable living space. |
| Fireplaces | Number of fireplaces. | Measures the presence and extent of this amenity. |
| FireplaceQu | Fireplace quality rating; 0 if no fireplace. | Distinguishes the value of existing fireplaces. |
| GarageArea | Garage area in square feet; 0 if no garage. | Captures parking and storage space. |
| GarageQual | Garage quality rating; 0 if no garage. | Captures quality beyond garage size. |
| GarageCond | Garage condition rating; 0 if no garage. | Captures maintenance needs beyond garage quality. |
| PavedDrive | Driveway surface category. | May affect convenience and reflect property upkeep. |
| PoolQC | Pool quality rating; 0 if no pool. | Tests whether a rare amenity adds predictive information. |
| Age | `max(0, YrSold - YearBuilt)`. | Measures the home's age at sale. |
| QualityAdjustedLivingArea | `GrLivArea × OverallQual`. | Allows the value of additional living area to vary with quality. |
| RenovationFreshness | `1 / (1 + max(0, YrSold - YearRemodAdd))`. | Higher values indicate a more recent renovation at sale. |

**Coding assumptions.** For the quality and condition fields with levels `Po/Fa/TA/Gd/Ex`, we propose scores `1/2/3/4/5`. This assumes that each one-level increase has the same modeled effect. Where `NA` specifically means that the facility does not exist, we assign 0, assuming absence is worth less than the lowest-quality existing facility, all else equal. A genuinely unknown value is not coded as 0. `BsmtExposure` uses its own ordered categories (`No=1, Mn=2, Av=3, Gd=4`, with 0 for no basement); `PavedDrive` remains categorical. We will compare the ordered-score approach with category dummies in validation. As required by the proposal instructions, qualitative variables will also be represented as dummies in the summary-statistics table.(Because there is a famous university-lowa State University in Ames, I may add "Neighborhood" as new variable. Dummy variable-whether this house in the area of Low State University)

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

## Notes on Missing Values and Categorical Features / 缺失值与定性变量说明

1. **Unknown basement exposure / 地下室采光未知。** One training house has a basement, but its `BsmtExposure` value is missing. This is an unknown rating, not “no basement.” We will fit an imputation rule using only the training portion of each cross-validation fold and apply it to that fold's validation portion. For the final model, we will fit the rule on the full training set and apply it to the competition test set. / 训练集中有一套房屋存在地下室，但 `BsmtExposure` 未记录；这表示采光等级未知，不表示没有地下室。每折交叉验证只用训练部分确定填补规则，再用于该折的验证部分；最终用完整训练集确定规则，再用于竞赛测试集。

2. **Categorical features / 定性特征。** For ordered quality and condition variables, we propose numeric scores (Po=1, Fa=2, TA=3, Gd=4, Ex=5; 0 only for a documented absence) to keep the initial model manageable. This assumes equal modeled differences between adjacent levels and that absence is below the lowest existing level. We will compare these scores with dummy encoding using cross-validation. Unordered categories such as `PavedDrive` will use dummies. The summary-statistics table will still show every qualitative variable as a full set of category dummies, as required; a regression with an intercept omits one reference dummy per variable. If we use feature scaling, we will fit its parameters on each fold's training portion only. / 对有顺序的质量和状况变量，我们拟采用数值编码（Po=1、Fa=2、TA=3、Gd=4、Ex=5；仅在确认设施不存在时记为0），使初始模型更简洁。这假设相邻等级在模型中的差距相等，且没有设施低于质量最差但仍存在的设施。我们会通过交叉验证与 dummy 编码比较。`PavedDrive` 等没有自然等级的类别仍使用 dummy。按照要求，描述性统计表仍将每个定性变量的所有类别分别列为 dummy；带截距的回归对每个变量省略一个基准 dummy。如使用特征缩放，也只用每折的训练部分计算缩放参数。