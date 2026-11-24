# Kaggle Competition / Econ 326

## 研究步骤 / Research workflow — 2026-11-24

| 步骤 / Step | 我们怎么做 / What we do | 解决什么问题？ / Why this step matters |
|---|---| --- |
| 1 · 定目标 / Set the goal | 用房屋信息预测未见过的房子的成交价，以更低的 log RMSE 为目标。 / Predict sale prices for unseen homes and aim for lower log RMSE. | 先说清楚“准”指什么，后面才能按同一目标选方案。 / A clear target tells us what counts as a better prediction. |
| 2 · 定比较方法 / Set up a fair test | 把 1,460 套房分成五份。每轮用 1,168 套训练，给剩下 292 套估价，轮换五次。各方案用相同分组，比较五轮的平均 log RMSE。填补缺失值和缩放只用当轮训练数据。 / Train on four groups and predict the fifth. Rotate five times. Use the same groups for every model, fit preprocessing on the training data and compare mean log RMSE. | 就算模型背下了全部成交价，拿原来那些房子测也能得满分，可第 1,461 套它估得准吗？轮换五次，每套房都有一次由没见过它成交价的模型来估价。这样能看出过拟合的迹象；五折本身不会治好过拟合。选好方案后，可以用全部房子重新训练。用 log RMSE 打分，两套房都高估 10%，对数误差就一样。 / A model could memorize every sale price and still miss on the next house. With five folds, every home gets one prediction from a model that never saw its price. This can reveal overfitting. Once the model is chosen, it can train on all the homes. With log RMSE, two prices overestimated by 10% have the same log error. |
| 3 · 看已有证据 / Start from the evidence | 11 月 22 日报告记录：面积、整体质量、房龄这 3 项得 0.1818，14 项得 0.1314。14 项是目前的一个参照，候选范围仍涵盖原始 79 个变量及其组合。 / The Nov 22 report gives 0.1818 for size, quality and age, and 0.1314 for 14 features. The 14-feature set is a reference point. All 79 original variables and their combinations remain candidates. | 先知道简单模型能做到什么，才看得出增加信息值不值得；14 项胜过 3 项，其他组合仍值得试。 / A baseline shows what extra information adds and leaves room for better combinations. |
| 4 · 改一个条件 / Change one thing | 下一步可以加变量、删变量，或把相关变量用 PCA 压缩后再回归。每次固定模型和折，只改特征方案。若试 PCA，在训练折内拟合并选择保留几个主成分。 / Try adding, removing or combining features. Keep the model and folds fixed. For PCA, fit the transformation and choose the component count within the training data. | 如果同时换模型、改特征，分数变了也说不清原因。一次改一项，才能看出这次改动有没有帮助。 / Changing one thing at a time shows whether that change helped. |
| 5 · 锁定后再验 / Test on fresh data | 同一批数据已经用来选特征，因此当前分数是开发阶段结果。锁定方案后，再用从未参与选择且有真实成交价的数据评估。PCA 对照和最终独立检验均待做。 / The current scores guided feature selection. Freeze the chosen approach, then test on fresh labeled sales. The PCA comparison and independent test are still pending. | 反复用同一批房子挑方案，可能碰巧挑中在这批房子上表现好的组合。用未参与选择的成交记录，才能检验优势能否延续。 / Repeated comparisons can favor a lucky result. Fresh sales test whether the gain carries over. |

## Updated comparison plan / 教授建议后的比较方案 — 2026-11-16

**研究重点：保留普通线性回归作为 benchmark，检验机器学习回归能否提高样本外预测准确度。**

- **Benchmark：** 用 OLS 估计的普通线性回归；具体特征和模型设定待确定。
- **机器学习对照方案：** 具体方法待确定；需要说明它与基准模型在设定或训练目标上的实质区别。
- **比较方式：** 两种模型均预测 ln(SalePrice)，使用相同的五折划分，比较平均验证 log RMSE；填补与 scaling 只根据每折训练部分确定。若需要调参，只在外层训练部分内完成，再由保留的验证折评分。
- **关键区别：** 相同特征、相同的无正则化平方误差目标下，OLS 和收敛后的梯度下降得到的预测基本一致；单独把 OLS 换成 gradient descent 不会自动提高准确度。
- **已有结果的含义：** 下方 A/B 结果比较的是 3 个与 18 个特征的两个线性模型，均由梯度下降拟合。9.42% 的改善来自这两种特征设定的比较，不能据此声称“机器学习胜过普通回归”。

Following the professor's advice, we will keep ordinary linear regression (OLS) as the benchmark and test whether a machine-learning regression approach improves out-of-sample accuracy. We will predict the natural logarithm of sale price and compare mean validation log RMSE using identical folds and training-only preprocessing. The exact benchmark specification and competing method remain to be chosen.


## Prediction target and cost / 预测目标与损失函数 — 2026-11-16

令 $y_i=\ln(\mathrm{SalePrice}_i)$，使用自然对数。线性模型预测对数房价：

$$
\hat y_i=f_{\vec{w},b}(\vec{x}_i)=\vec{w}\cdot\vec{x}_i+b.
$$

$$
J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}\left(f_{\vec{w},b}(\vec{x}_i)-\ln(\mathrm{SalePrice}_i)\right)^2.
$$

梯度下降对每个权重计算 $\partial J/\partial w_i$，并更新截距；完整公式见 [outline](outline.md#proposed-regression-model)。18 个 features 的定义不变，不因目标取 log 而全部取 log。美元预测值可由 $\exp(\hat y_i)$ 得到。

**状态：文档已改为 log 目标；现有训练脚本、系数与验证结果仍属于旧的美元目标，尚未按 log 目标重新训练。** 以下历史数字不能作为新模型的 log RMSE。

## Five-fold cross-validation / 五折交叉验证 — 2026-11-11

**本次比较中，18 特征模型 B 的平均验证 RMSE 最低。** 两个模型使用同一组五折划分，每折 1,168 套训练、292 套验证；填补规则和缩放参数均只根据该折训练数据确定。

| 模型 | 特征 | 平均验证 RMSE（美元） |
|---|---|---:|
| A：基础模型 | GrLivArea、OverallQual、Age | 40,701.83 |
| B：完整模型 | 当前 18 个特征 | **36,865.74** |
| 均价基准 | 每折只预测其训练部分的平均房价 | 79,273.77 |

B 比 A 降低 **3,836.09 美元（9.42%）**，但第 3 折更差、跨折波动更大。这里比较了两种预先确定的特征组合，并未证明每个新增特征都有帮助。

**[查看每折结果、处理规则和代码](results/five_fold_cv/README.md)**。在当时的美元目标比较中，按最低平均验证 RMSE 规则选择 B；下方已有 B 在全体 1,460 套房屋上重新训练的系数。


## Gradient descent results / 梯度下降结果 — 2026-11-11

已用全部 **1,460 套房屋、现有 18 个数值特征**完成一次全量训练，直接预测美元售价。系数从 0 开始，通过梯度下降更新。

- **[查看系数表和训练记录](results/gradient_descent/README.md)**：包括可代入原始特征的截距与 w1–w18，以及缩放后的另一套系数。
- 最终训练 cost（MSE / 2）：**609,081,328.15 美元²**；训练 RMSE：**34,902.19 美元**。
- 学习率 **0.180802912458**；迭代 **134,062 次**。梯度收敛、与独立最小二乘解的比较、缩放前后预测一致性检查均通过。
- **[运行代码](scripts/fit_gradient_descent.py)**；**[完整精度系数和预处理参数](results/gradient_descent/model.json)**。
- 以上仍是全量训练误差；**样本外的五折验证结果见上方**。未来的 NearISU 尚未构造或加入。

## Earlier plan / 原方案记录 — 2026-11-11

**当时的方案：直接预测 `SalePrice`（美元），用平方误差 cost 和梯度下降估计系数，以五折交叉验证的美元 RMSE 比较模型。**

1. **Feature selection：避免 over-controlling（过度控制）。** 选择控制变量时，要明确“保持其他变量不变”是否符合研究问题，避免因控制不恰当的变量而改变系数的含义。对于本项目的预测目标，仍以交叉验证的样本外误差比较 feature 方案；预测关系不自动代表因果作用。
2. **Feature encoding：按现有方案编码。** 18 个候选特征均作为数值输入：有序质量与状况等级、地下室采光和车道铺装按 `outline.md` 中的规则评分。未来加入 `Neighborhood` 时，可用社区类别 dummy 与现有模型比较；模型包含截距时，K 个类别通常使用 K−1 个 dummy，并保留一个基准类别。Dummy 输入保持为 0/1；训练根据真实售价估计其系数。在没有交互项时，该系数表示相对基准组的预测售价差异，单位为美元。

## To-do list / 项目待办

- [ ] **1. 固定特征定义。** 列出当前 18 个候选特征的计算与编码规则，核对 `train.csv`、`test.csv` 的列名，并检查缺失值和异常年份。
- [x] **2. 建立预处理流程。** 在每一折训练集上确定未知值的填补规则和数值特征的 scaling 参数，再应用到该折验证集，避免使用验证集信息。
- [ ] **3. 确定新的比较模型。** 明确 OLS benchmark 的特征和设定，再选择有实质区别的机器学习回归方案；在评估前固定比较规则。
- [x] **4. 完成原美元目标的梯度下降。** 已完成当前 18 特征模型的全量训练，见上方结果记录。以 `SalePrice`（美元）为目标，写出预测函数、平方误差 cost 和每个权重及截距的更新；选择初始学习率，确认训练 cost 随迭代下降并趋于稳定。
- [x] **5. 完成原 A/B 五折比较。** 已比较 3 特征与 18 特征线性模型，平均验证 RMSE 分别为 40,701.83 和 36,865.74 美元。这是已有的特征设定比较。
- [ ] **4a. 改为 log 目标并重新训练。** 更新训练与交叉验证代码中的目标为 ln(SalePrice)，重新检查梯度收敛和最小二乘解，单独保存新系数、cost 与 log RMSE，不覆盖美元目标的历史结果。
- [ ] **5a. 执行教授建议的新比较。** 确定两种方案后，在相同外层验证折上比较 OLS benchmark 与机器学习模型；调参仅使用外层训练部分，记录每折 log RMSE、平均值及波动。
- [ ] **6. 整理结果与解释。** 报告各模型的 log RMSE、加入特征后的变化，并说明数值等级编码、异常值处理及局限。
- [ ] **7. 最终训练与预测。** 用全部 1,460 条带价格数据重新拟合选定流程；模型输出对数售价，需要美元预测时取 exp。若另做竞赛预测，再应用到 1,459 条无售价测试数据；课程项目的准确性根据带标签数据的验证折评估。
- [ ] **8. 完成提交文档。** 使 research statement、变量表、统计表、回归方程与实际使用的模型一致；完成 Jupyter notebook 与简短报告，核对篇幅、引用和课程提交要求；确认已与教师讨论进阶方向的研究方案。
