# gyw 分支代码与结果 Review 报告

审查时间：2026-05-15  
审查分支：`gyw`  
合并目标分支：`codex/merge-codex-gyw-review`  
基准分支：`codex/preprocessing-modeling`

## 一、合并状态

已创建新分支 `codex/merge-codex-gyw-review`，并将当前 Codex 分支与 `gyw` 分支合并。

合并时仅出现 `requirements.txt` 冲突。已处理为统一依赖文件，保留 Codex 线依赖，并加入 gyw 线频繁模式挖掘所需的 `mlxtend`、`ipykernel`、`nbconvert`。

合并后主要新增内容包括：

- `notebooks/01_eda.ipynb`
- `notebooks/05_outlier_analysis.ipynb`
- `notebooks/06_frequent_patterns.ipynb`
- `outputs/figures/01_*`、`05_*`、`06_*`
- `outputs/tables/01_feature_stats.csv`
- `outputs/tables/woe_iv_top20.csv`
- `outputs/tables/06_frequent_itemsets.csv`
- `outputs/tables/06_association_rules.csv`
- `docs/` 下的方案、分工和项目资料

## 二、主要发现

### P1：`anomaly_score_*.npy` 未纳入结果，导致 Codex 消融实验 E5/E6 无法完成

`notebooks/05_outlier_analysis.ipynb` 的说明和代码都声明会输出 `../outputs/anomaly_score_train.npy` 与 `../outputs/anomaly_score_valid.npy`，代码中也调用了 `np.save(...)` 保存这两个文件。但合并后的 `outputs/` 目录并没有这两个文件。

影响：

- `notebooks/03_lightgbm_model.ipynb` 中 E5、E6 已经因为缺少 `anomaly_score_*.npy` 被跳过。
- 分工文档中唯一跨线依赖没有真正交付，Codex 线无法完成“原始 + IsolationForest anomaly_score”和“全部叠加”的消融实验。
- 报告表 2 若写入 E5/E6，会出现缺项或不可复现。

建议：

- 重新运行 `notebooks/05_outlier_analysis.ipynb`，确认生成并提交：
  - `outputs/anomaly_score_train.npy`
  - `outputs/anomaly_score_valid.npy`
- 之后重新运行 `notebooks/03_lightgbm_model.ipynb`，补齐 E5/E6 消融实验。

### P1：LOF 实现使用全量 160k 训练样本，偏离分工要求且存在明显性能风险

分工要求 LOF 随机抽取 30000 条样本，避免 LOF 的近邻计算在全量数据上成本过高。但 `notebooks/05_outlier_analysis.ipynb` 当前直接对 `X_train_s` 全量训练集运行：

```python
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
lof_labels = lof.fit_predict(X_train_s)
```

影响：

- 与分工文档要求不一致。
- 在普通笔记本或云端环境中可能运行非常慢，甚至内存不足。
- 如果报告写“按分工抽样 30000”，但代码实际全量运行，会造成复现口径不一致。

建议：

- 使用 `np.random.default_rng(42).choice(...)` 从训练集抽取 `min(30000, len(X_train_s))` 条样本运行 LOF。
- 图表和结论中明确标注 LOF 是基于 30000 样本抽样结果。

### P2：频繁模式挖掘参数与分工文档不一致，当前结果更像“降维后的探索性规则”

分工要求：

- 对 200 个特征做 3 档等频分箱。
- FP-Growth 使用 `min_support=0.3`。
- 关联规则使用 `min_lift=1.2`。

当前 `notebooks/06_frequent_patterns.ipynb` 实际实现为：

- 先按正负样本均值差异筛选 Top 30 特征。
- `MIN_SUPPORT = 0.05`
- `MIN_LIFT = 1.01`

输出结果中，`outputs/tables/06_association_rules.csv` 的 Top 规则 lift 约为 `1.04~1.06`，支持度约为 `0.116~0.117`。

影响：

- 这套结果是可以解释的探索性分析，但不等同于原分工约定的强规则挖掘。
- 如果报告直接写“按 200 个特征、min_support=0.3、min_lift=1.2 完成”，会与代码和结果不一致。
- 当前规则 lift 非常接近 1，只能说明弱共现，不能写成“显著强关联”。

建议：

- 报告中明确说明：由于 Santander 特征近似独立，严格参数下难以产生有解释性的多项规则，因此实际采用 Top 30 特征、`min_support=0.05`、`min_lift=1.01` 作为探索性规则挖掘。
- 若必须完全对齐分工，则需要补跑 200 特征版本，并记录“严格阈值下规则稀少或无强规则”的负结果。

### P2：`pyproject.toml` 与 `requirements.txt` 环境约束不一致，可能阻断 uv 环境复现

`pyproject.toml` 中要求：

- `requires-python = ">=3.12"`
- `numpy>=2.4.4`
- `pandas>=3.0.3`
- `scikit-learn>=1.8.0`
- `ipykernel>=7.2.0`

而本项目当前可运行环境使用的是系统 `python3.9`，并且 `requirements.txt` 已合并为更宽松、当前可安装的版本下限。

影响：

- 使用 `pip install -r requirements.txt` 可以走通当前 notebook 依赖。
- 使用 `uv sync` 或基于 `pyproject.toml` 建环境时，可能因为版本约束过高或 Python 版本要求不同而失败。
- 同一仓库存在两套不一致的依赖入口，会降低复现稳定性。

建议：

- 将 `pyproject.toml` 的依赖版本与 `requirements.txt` 对齐，或删除不使用的 `pyproject.toml` / `uv.lock`。
- 如果保留 uv 工作流，应重新生成 `uv.lock`。

### P3：EDA 输出完整，但部分扩展分析超出原分工，需要在报告中分清“必选结果”和“增强结果”

`notebooks/01_eda.ipynb` 完成了样本规模、类别分布、特征统计、Top 20 KDE、Top 30 相关性热力图等分工要求。额外加入了：

- Top 10 特征分箱正例率折线图。
- WoE / IV 分析。
- `outputs/tables/woe_iv_top20.csv`。

这些增强分析是有价值的，但 IV / WoE 并不是原方案核心要求。报告中建议作为补充特征区分度分析，避免喧宾夺主。

## 三、结果核对

### EDA

已生成：

- `outputs/figures/01_target_distribution.png`
- `outputs/figures/01_kde_top20.png`
- `outputs/figures/01_corr_heatmap_top30.png`
- `outputs/figures/01_corr_top20_bar.png`
- `outputs/figures/01_mean_diff_top20.png`
- `outputs/figures/01_binning_positive_rate.png`
- `outputs/figures/01_iv_top20.png`
- `outputs/tables/01_feature_stats.csv`
- `outputs/tables/woe_iv_top20.csv`

结果可用于报告第 2 章。

### 离群检测

已生成：

- `outputs/figures/05_isolation_forest_analysis.png`
- `outputs/figures/05_lof_analysis.png`
- `outputs/figures/05_train_test_anomaly_distribution.png`

缺失：

- `outputs/anomaly_score_train.npy`
- `outputs/anomaly_score_valid.npy`

离群检测结果可以用于报告第 5 章，但跨线消融输入未交付。

### 频繁模式挖掘

已生成：

- `outputs/figures/06_association_rules_scatter.png`
- `outputs/figures/06_pos_neg_support_diff.png`
- `outputs/figures/06_top20_frequent_itemsets.png`
- `outputs/figures/06_top20_rules_lift.png`
- `outputs/tables/06_frequent_itemsets.csv`
- `outputs/tables/06_association_rules.csv`

结果表规模：

- `06_frequent_itemsets.csv`：4005 行。
- `06_association_rules.csv`：保存 Top 50 条规则。
- Top 规则 lift 约为 `1.056`，说明规则强度较弱。

结果可用于报告第 5 章，但必须按“弱关联探索”表述。

## 四、总体评价

`gyw` 分支整体覆盖了分工中“我的线”的主要模块：EDA、离群检测、频繁模式挖掘，并生成了多数图表和表格。EDA 部分完成度较高，图表足够支撑报告；频繁模式部分做了合理的工程降维，但参数与原分工不一致，需要在报告中解释；离群检测部分最大问题是缺少 `.npy` 交付物，并且 LOF 全量运行偏离分工。

建议优先处理顺序：

1. 补齐 `anomaly_score_train.npy` 和 `anomaly_score_valid.npy`，再重跑 Codex 线 E5/E6。
2. 将 LOF 改为 30000 样本抽样。
3. 统一 `pyproject.toml` 与 `requirements.txt`。
4. 在最终报告中明确频繁模式挖掘采用的是 Top 30 特征、低阈值探索策略，避免与分工文档冲突。
