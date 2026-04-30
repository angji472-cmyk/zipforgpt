# Group-Aware EEG Emotion Recognition Design Note

## 0. 当前项目背景

当前主线模型已经形成较稳定方案：

- 主线：Conformer + SRFNet rank ensemble
- 解码策略：subject-level Top-4 decoding
- 当前 best：BA = 0.7958
- NoTop4 固定阈值 0.5 明显弱于 Top-4
- EA-SRFNet smoke test 未通过，且 NoTop4 性能大幅下降

已有诊断表明：

| 方法 | Mean BA |
|---|---:|
| Top-4 current best | 0.7958 |
| Fixed Threshold 0.5 | 0.7812 |

因此当前任务强依赖 subject-level Top-4 结构先验。

EA 实验中，`subject_self + raw_branch_only` 在 fold0 上：

| 指标 | 原始 SRFNet | EA-SRFNet | 变化 |
|---|---:|---:|---:|
| Top4 BA | 0.7708 | 0.7500 | -0.0208 |
| NoTop4 BA | 0.7812 | 0.7083 | -0.0729 |

这说明 EA 可能破坏了底层 score quality 和概率校准能力。当前不建议继续把 EA 纳入主线训练。

---

## 1. 为什么考虑抑郁 / 正常分组设计？

本项目被试存在两类：

- depressed / 抑郁组
- normal / control / 正常组

这不是普通随机 subject 差异，而可能对应真实生理差异。

抑郁组与正常组在 EEG 情绪识别中可能存在：

| 层面 | 可能差异 |
|---|---|
| 基线脑电 | 静息态功率谱、左右脑不对称、前额叶活动可能不同 |
| 情绪诱发反应 | 同样刺激下情绪反应幅度可能不同 |
| 通道协方差 | 脑区间耦合模式可能不同 |
| 概率校准 | 模型对某一组输出分数可能整体偏高或偏低 |
| Top-4 排序 | 某一组 subject 内 trial 排序可能更难 |

因此，相比继续做强归一化，当前更值得做：

```text
group-aware diagnosis
→ group-specific ensemble weight
→ group-aware calibration
→ group-conditioned lightweight model
```

不要一开始就做大规模结构修改。

---

## 2. 核心原则

### 2.1 不改主线训练代码

第一阶段只做分析和后处理：

- 不重新训练主模型；
- 不改 Conformer / SRFNet 主线；
- 不改变 subject-level Top-4 解码；
- 不使用 public feedback 或 test label 调参；
- 不引入任何标签泄漏。

### 2.2 先诊断，再设计

不要预设“抑郁组一定需要特殊模型”。

必须先回答：

1. 当前模型在 depressed / normal 上是否性能差异明显？
2. fold3 弱是否集中在某一组？
3. Conformer 和 SRFNet 是否在不同 group 上互补？
4. group-specific ensemble weight 是否能稳定提升？
5. 提升是否跨 fold 稳定，而不是单 fold 过拟合？

---

## 3. Phase G0：确认 group 标签

### 目标

确认每个 subject 是否有可靠 group 标签。

### 输入候选

检查以下位置是否存在 subject group 信息：

- manifest
- metadata
- subject info csv/xlsx/json
- split manifest
- 原始数据说明文件
- 项目 README / notebook / config

### 输出文件

生成：

```text
outputs/group_analysis/group_subject_audit.csv
outputs/group_analysis/group_distribution_by_fold.csv
```

### `group_subject_audit.csv` 字段

| 字段 | 说明 |
|---|---|
| subject_id | 被试 ID |
| group | depressed / normal |
| fold | 所属 fold |
| split | train / val / test |
| n_trials | trial 数 |
| n_positive | 正类 trial 数 |
| n_negative | 负类 trial 数 |
| source | group 标签来源 |

### 检查点

如果找不到 group 标签：

```text
STOP_GROUP_ANALYSIS
reason = "No reliable subject group label found."
```

不要猜测 subject 属于哪一类。

---

## 4. Phase G1：现有主线分组诊断

### 目标

评估当前 best ensemble 在 depressed / normal 两组上的表现。

### 输入

读取当前 best ensemble 的预测结果，要求至少包含：

| 字段 | 说明 |
|---|---|
| subject_id | 被试 ID |
| fold | fold 编号 |
| trial_id / original_trial_id | 原始 trial ID |
| y_true | 真实标签 |
| prob / score | 模型分数 |
| y_pred_top4 | Top-4 预测 |
| group | depressed / normal |

如果 group 不在预测文件中，则从 G0 的 `group_subject_audit.csv` merge。

### 统计指标

按 group 统计：

| 指标 | 说明 |
|---|---|
| Top4 BA | Top-4 后 balanced accuracy |
| Top4 Macro-F1 | Top-4 后 macro F1 |
| NoTop4 BA | 固定阈值 0.5 的 BA |
| AUC | 原始分数 AUC |
| mean(prob) | 分数均值 |
| std(prob) | 分数标准差 |
| pos_mean(prob) | 正类平均分 |
| neg_mean(prob) | 负类平均分 |
| margin | pos_mean - neg_mean |
| subject_top4_hit | subject 内 Top-4 命中情况 |

### 输出文件

```text
outputs/group_analysis/group_metrics_summary.csv
outputs/group_analysis/group_subject_metrics.csv
outputs/group_analysis/group_score_distribution.csv
outputs/group_analysis/group_fold_breakdown.csv
```

### 重点判断

如果结果类似：

```text
depressed BA << normal BA
```

说明模型对抑郁组泛化更差，可以继续 group-aware 设计。

如果结果类似：

```text
depressed BA ≈ normal BA
```

说明分组设计收益可能有限，不建议改主线。

---

## 5. Phase G2：Group-Specific Ensemble Weight Sweep

### 目标

不训练模型，只在 score 层做 group-specific ensemble。

当前主线是 Conformer + SRFNet ensemble。假设已有：

```text
conformer_score
srfnet_score
subject_id
group
fold
y_true
```

对 depressed 和 normal 使用不同融合权重：

```python
if group == "depressed":
    final_score = w_dep * conformer_score + (1 - w_dep) * srfnet_score
else:
    final_score = w_norm * conformer_score + (1 - w_norm) * srfnet_score
```

然后仍然对每个 subject 执行 Top-4：

```text
for each subject:
    select top 4 trials by final_score as positive
```

### 权重搜索范围

```python
w_dep_list = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
w_norm_list = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
```

也可以额外做细扫：

```python
w_dep_list = [0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
w_norm_list = [0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]
```

但细扫必须在粗扫有稳定提升后再做。

### 输出文件

```text
outputs/group_analysis/group_weight_sweep.csv
outputs/group_analysis/best_group_weight_predictions.csv
outputs/group_analysis/best_group_weight_summary.json
```

### `group_weight_sweep.csv` 字段

| 字段 | 说明 |
|---|---|
| w_dep | depressed 组 Conformer 权重 |
| w_norm | normal 组 Conformer 权重 |
| overall_ba | 总 BA |
| overall_mf1 | 总 Macro-F1 |
| depressed_ba | 抑郁组 BA |
| normal_ba | 正常组 BA |
| fold0_ba | fold0 BA |
| fold1_ba | fold1 BA |
| fold2_ba | fold2 BA |
| fold3_ba | fold3 BA |
| fold4_ba | fold4 BA |
| n_folds_improved | 相比 baseline 提升的 fold 数 |
| decision | PASS / FAIL |

---

## 6. Phase G3：防过拟合检查

### 当前 baseline

```text
baseline_overall_ba = 0.7958
```

### 通过标准

group-specific ensemble 只有满足以下条件才可进入候选：

```text
overall BA > 0.7958
AND 至少 3/5 folds 不下降
AND depressed / normal 不能有任意一组明显崩掉
AND fold3 不能靠牺牲其他 fold 换取单点提升
AND 不使用 public feedback / private test label
```

推荐判定逻辑：

```python
pass_overall = overall_ba > 0.7958
pass_folds = n_folds_not_worse >= 3
pass_group = min(depressed_ba_delta, normal_ba_delta) > -0.02

if pass_overall and pass_folds and pass_group:
    decision = "GROUP_ENSEMBLE_CANDIDATE"
else:
    decision = "DIAGNOSTIC_ONLY"
```

---

## 7. Phase G4：可选的 Group-Aware Calibration

如果 G2 没有提升，但诊断发现某一组 score calibration 明显偏移，可以尝试轻量校准。

### 不推荐方案

如果最终仍然使用 subject-level Top-4，那么对同一 subject 内做：

```text
score_adj = a_group * score + b_group
```

通常不会改变排序，因此对 Top-4 结果没有帮助。

### 推荐方案

只在 ensemble 融合前做 group-aware calibration：

```python
conformer_score_calibrated = calibrate_by_group(conformer_score, group)
srfnet_score_calibrated = calibrate_by_group(srfnet_score, group)
final_score = weighted_ensemble(conformer_score_calibrated, srfnet_score_calibrated)
```

可以尝试：

```text
group-wise z-score
group-wise temperature scaling
group-wise rank normalization
```

但必须防止 label leakage。

优先使用 out-of-fold statistics，不要用 test label 拟合校准参数。

---

## 8. Phase G5：可选的 Group-Conditioned Model

只有当 G1 / G2 明确显示 group-aware 后处理有稳定价值时，才考虑动模型。

### 方案 A：Group Embedding + FiLM

结构：

```text
EEG input
  ↓
shared backbone
  ↓
feature h
  ↓
group embedding
  ↓
gamma_group, beta_group
  ↓
h' = gamma_group * h + beta_group
  ↓
classifier
```

优点：

- 参数量小；
- 可以建模抑郁 / 正常组的特征偏移；
- 不会像 EA 一样强行白化协方差。

风险：

- subject 数少时容易过拟合；
- 如果测试时 group 标签不可获得，则不能使用。

### 方案 B：Shared Backbone + Group-Specific Head

结构：

```text
EEG input
  ↓
shared backbone
  ↓
shared feature
  ↓
normal head / depressed head
```

训练 loss：

```python
loss = ce_loss(shared_head_logits, y) + lambda_group * ce_loss(group_head_logits, y)
```

推理时：

```python
if group == "depressed":
    use depressed_head
else:
    use normal_head
```

风险：

- 数据少时 group head 不稳定；
- 可能只记住 fold 内 subject 偏差。

### 方案 C：Group-Balanced Sampler

如果 depressed / normal 数量不平衡，可以做：

```text
每个 batch 尽量包含相近数量 depressed / normal subject
```

或者：

```python
loss = group_weight[group] * ce_loss
```

这比改结构更安全。

---

## 9. 不建议优先做 Group-Specific EA

EA 的失败已经说明：

```text
subject-level covariance whitening may destroy useful discriminative structure.
```

如果继续做 EA，只能作为低优先级探索。

### 不推荐

```text
直接对 depressed / normal 分别做 full EA whitening
```

因为这可能继续抹掉两组之间有意义的 EEG 协方差差异。

### 如果必须尝试

只能做 soft EA：

```python
T_soft = (1 - alpha) * I + alpha * R_group^{-1/2}
```

搜索：

```python
alpha_list = [0.1, 0.2, 0.3]
```

并且只做 smoke test，不进入主线。

通过标准必须严格：

```text
Top4 BA 不下降
NoTop4 BA 不显著下降
至少 3/5 folds 稳定
```

---

## 10. 最终推荐路线

优先级如下：

```text
P0: Group label audit
P1: Existing best ensemble group diagnosis
P2: Group-specific Conformer/SRFNet ensemble weight sweep
P3: Group-aware score calibration
P4: Group-conditioned lightweight head / FiLM
P5: Soft group-EA smoke test
```

当前最值得执行的是：

```text
Group-specific Conformer/SRFNet ensemble weight sweep + subject-level Top-4
```

原因：

- 不改训练；
- 不破坏主线；
- 不动 EEG 原始信号；
- 不抹掉协方差结构；
- 能利用抑郁 / 正常这个真实生理分组；
- 失败成本很低；
- 成功后可直接作为候选提交策略。

---

## 11. 给自动化 Agent 的任务摘要

请实现一个独立分析脚本，不改原训练代码。

脚本功能：

1. 找到 subject 的 depressed / normal group 标签；
2. 合并当前 best ensemble 的 OOF / fold 预测；
3. 按 group 输出性能诊断；
4. 对 Conformer 和 SRFNet 分数做 group-specific ensemble weight sweep；
5. 每组可用不同 ensemble 权重；
6. 最终仍然执行 subject-level Top-4；
7. 输出完整 csv/json 报告；
8. 严格检查是否超过 baseline BA=0.7958；
9. 如果没有稳定提升，只输出 DIAGNOSTIC_ONLY，不改主线。

最终输出目录：

```text
outputs/group_analysis/
```

必须生成：

```text
group_subject_audit.csv
group_distribution_by_fold.csv
group_metrics_summary.csv
group_subject_metrics.csv
group_score_distribution.csv
group_fold_breakdown.csv
group_weight_sweep.csv
best_group_weight_predictions.csv
best_group_weight_summary.json
group_analysis_final_report.md
```

最终报告必须包含：

```text
1. group 标签来源
2. depressed / normal subject 数量
3. 各 fold group 分布
4. 当前 best 在两组上的表现
5. 最优 group-specific ensemble 权重
6. 是否超过 BA=0.7958
7. 是否至少 3/5 folds 不下降
8. depressed / normal 是否有一组被牺牲
9. 最终决策：
   - GROUP_ENSEMBLE_CANDIDATE
   - or DIAGNOSTIC_ONLY
```
