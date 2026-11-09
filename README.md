# Kaggle Competition / Econ 326
## Update — 2026-10-28

1. **Feature selection：避免 over-controlling（过度控制）。** 选择控制变量时，要明确“保持其他变量不变”是否符合研究问题，避免因控制不恰当的变量而改变系数的含义。对于本项目的预测目标，仍以交叉验证的样本外误差比较 feature 方案；预测关系不自动代表因果作用。
2. **Categorical features：编码为 dummy variables 后，与其他 features 一起放入同一个 regression model。** Dummy 的输入值保持为 0/1；训练时根据 training set 的真实房价（本项目拟合 log(SalePrice)），通过最小化误差估计各变量的系数，而不是调整 dummy 的 0/1 值。没有交互项时，dummy 的系数表示相对基准组的预测水平偏移（shift，在本项目中是 log 房价的偏移）；加入交互项后，这个差异可以随其他 feature 改变。模型包含截距时，具有 K 个类别的变量通常使用 K−1 个 dummy，并保留一个基准类别，避免完全共线性。
