# AI 下一步任务：对齐原始 EEGNET exp3_vote_5fold 与 AdaBN fork，并验证 AdaBN 是否真实有效

## 任务背景

当前 AdaBN 消融实验已经完成，但结果显示：

- 原始 EEGNET `exp3_vote_5fold`：
  - BA = 0.6329±0.0191
  - Macro-F1 = 0.6285±0.0232
  - ROC-AUC = 0.6883±0.0211

- `EEGNET+AdaBN+DANN` fork 中的 A — AdaBN-off control：
  - BA = 0.6042±0.0214
  - Macro-F1 = 0.5937±0.0278
  - ROC-AUC = 0.6733±0.0301

- `EEGNET+AdaBN+DANN` fork 中的 B — AdaBN train-only：
  - BA = 0.6242±0.0394
  - Macro-F1 = 0.6148±0.0434
  - ROC-AUC = 0.6822±0.0273

- `EEGNET+AdaBN+DANN` fork 中的 C — AdaBN val-transductive：
  - BA = 0.6221±0.0298
  - Macro-F1 = 0.6052±0.0316
  - ROC-AUC = 0.6896±0.0330

关键问题：

> A — AdaBN-off control 没有复现原始 `exp3_vote_5fold`，差距约 BA -0.0287、Macro-F1 -0.0348。  
> 因此，目前不能直接认为 AdaBN 对原始最优模型有效。

本任务的目标是：

1. 找出原始 `EEGNET/outputs/exp3_vote_5fold` 和 fork 中 `adabn_off_exp3_vote_5fold` 的差异；
2. 让 AdaBN-off control 尽可能复现原始 `exp3_vote_5fold`；
3. 在复现成功后，再测试 AdaBN train-only 是否有真实增益；
4. 暂停 DANN，不要实现 DANN。

---

## 0. 总体原则

1. 不要删除任何文件。
2. 不要覆盖已有实验结果。
3. 不要修改原始 `exp3_vote_5fold` 结果目录。
4. 不要继续实现 DANN。
5. 当前任务只做：
   - 差异审计；
   - baseline 复现；
   - 原始 EEGNET 路径下的 AdaBN train-only / val-transductive 验证。
6. 所有新输出统一放入：

```text
EEGNET/outputs_adabn_alignment/
```

7. 所有报告和 CSV 使用相对路径。
8. 所有实验必须保存：
   - config snapshot
   - run command
   - random seed
   - fold split hash
   - train/val subject list
   - metrics json
   - prediction csv
   - leakage check json
   - run_metadata.json

9. public 数据不得参与训练、调参、early stopping、模型选择或指标计算。
10. 如果使用 validation 无标签数据更新 BN，必须标注为 `transductive`，不能进入普通 inductive leaderboard。
11. 如果发现 subject / trial / window 泄露，立即停止并报告。

---

## 1. 第一阶段：差异审计

请先不要跑新实验，先完成差异审计。

目标：

> 解释为什么 fork 中的 A — AdaBN-off control 比原始 `exp3_vote_5fold` 低约 0.0287 BA。

---

### 1.1 配置差异审计

比较以下配置：

```text
EEGNET/configs/exp3_dual_vote_alpha05.yaml
EEGNET+AdaBN+DANN/configs/adabn_off_exp3_vote_5fold.yaml
```

如果第二个配置文件路径不同，请在 `EEGNET+AdaBN+DANN/configs/` 中查找实际用于 A 实验的配置。

重点比较字段：

```text
model
model.type
fusion_type
alpha
input_shape
num_classes
batch_size
epochs
optimizer
learning_rate
weight_decay
scheduler
scheduler.type
ReduceLROnPlateau 参数
early_stopping
patience
seed
data_root
dataset
preprocessing
normalization
trial length
window length
sampling rate
label mapping
class weights
loss function
checkpoint selection
metric scope
fold split source
```

输出文件：

```text
EEGNET/outputs_adabn_alignment/config_diff_exp3_vs_adabn_off.md
```

报告中必须列出：

| 字段 | 原始 EEGNET exp3 | AdaBN fork A | 是否一致 | 可能影响 |
|---|---|---|---|---|

---

### 1.2 代码路径差异审计

比较两个项目中的关键代码文件：

```text
EEGNET/models/dual_branch_eegnet.py
EEGNET+AdaBN+DANN/models/dual_branch_eegnet.py

EEGNET/models/fft_layer.py
EEGNET+AdaBN+DANN/models/fft_layer.py

EEGNET/models/bandpower_layer.py
EEGNET+AdaBN+DANN/models/bandpower_layer.py

EEGNET/data/preprocessing.py
EEGNET+AdaBN+DANN/data/preprocessing.py

EEGNET/data/dataset.py
EEGNET+AdaBN+DANN/data/dataset.py

EEGNET/training/trainer.py
EEGNET+AdaBN+DANN/training/trainer.py

EEGNET/evaluation/
EEGNET+AdaBN+DANN/evaluation/
```

重点检查：

```text
DualBranchEEGNet forward 是否一致
FFT 分支是否一致
概率平均公式是否一致
alpha 是否一致
softmax/log-prob 处理是否一致
BN/dropout 行为是否一致
normalization 是否一致
train/eval mode 是否一致
loss 输入是 logits 还是 log-prob
early stopping 是否一致
best checkpoint 选择是否一致
DataLoader shuffle/drop_last 是否一致
random seed 是否一致
fold split 是否一致
evaluation aggregation 是否一致
```

输出文件：

```text
EEGNET/outputs_adabn_alignment/codepath_diff_exp3_vs_adabn_off.md
```

报告中必须明确：

1. 两边模型 forward 是否完全一致；
2. 两边训练 loop 是否完全一致；
3. 两边评估脚本是否完全一致；
4. 哪些差异最可能导致 0.0287 BA 差距。

---

### 1.3 Fold split 一致性审计

比较：

```text
EEGNET/outputs/exp3_vote_5fold
EEGNET+AdaBN+DANN/outputs_adaptation_ablation/adabn_off_exp3_vote_5fold
```

如果 fork 的输出路径不同，请自动查找对应的 A 实验目录。

对每个 fold 检查：

```text
train subject list
val subject list
test subject list
train sample_id list
val sample_id list
test sample_id list
class distribution
n_train
n_val
n_test
split_hash
```

输出文件：

```text
EEGNET/outputs_adabn_alignment/split_diff_exp3_vs_adabn_off.csv
```

CSV 字段：

```csv
fold,original_split_hash,fork_split_hash,same_split,n_train_original,n_train_fork,n_val_original,n_val_fork,class_dist_original,class_dist_fork,subject_overlap_diff,notes
```

如果 split 不一致，优先使用原始 `exp3_vote_5fold` 的 split 重新跑后续实验。

---

### 1.4 指标口径一致性审计

确认两个实验是否都使用：

```text
trial10s / all:all
```

检查：

```text
prediction aggregation
window 到 trial 的聚合方式
trial10s / trial50s 是否混用
threshold
positive class
balanced_accuracy 计算
macro_f1 计算
ROC-AUC 计算
是否使用 probability 而不是 hard label 计算 AUC
是否排除 public y_true=-1
```

输出文件：

```text
EEGNET/outputs_adabn_alignment/metric_scope_diff_exp3_vs_adabn_off.md
```

报告中必须回答：

1. 两边是否使用同一指标口径；
2. 两边 prediction CSV 的字段是否一致；
3. 两边 fold mean/std 计算方式是否一致；
4. 是否存在 trial/window 聚合差异。

---

## 2. 第二阶段：复现实验 R0

只有完成第一阶段后，才开始跑复现实验。

---

### 2.1 实验 R0：原始路径复跑

在原始 EEGNET 项目中复跑：

```text
exp3_vote_5fold_repro
```

输出目录：

```text
EEGNET/outputs_adabn_alignment/exp3_vote_5fold_repro
```

要求：

1. 使用原始 `EEGNET/configs/exp3_dual_vote_alpha05.yaml`。
2. 使用原始 `exp3_vote_5fold` 的 fold split。
3. 使用原始 EEGNET 训练路径。
4. 保存完整 metadata。
5. 使用 trial10s / all:all 评估口径。
6. 不使用 public 数据。
7. 不启用 AdaBN。
8. 不修改原始 `exp3_vote_5fold` 目录。

复现判定：

```text
BA 误差 <= 0.010
Macro-F1 误差 <= 0.010
```

如果 R0 不能复现原始结果，停止，不要继续跑 R1/R2，先输出失败原因。

输出：

```text
EEGNET/outputs_adabn_alignment/repro_vs_original.csv
```

字段：

```csv
experiment,ba_mean,ba_std,macro_f1_mean,macro_f1_std,roc_auc_mean,roc_auc_std,delta_ba,delta_macro_f1,reproduces_original,notes
```

---

## 3. 第三阶段：在原始 EEGNET 路径启用 AdaBN

只有 R0 复现成功后，才执行 R1/R2。

---

### 3.1 给原始 EEGNET 添加最小 AdaBN 支持

在原始 EEGNET 项目中实现最小 AdaBN 功能，尽量复用 fork 中已经验证过的：

```text
recalibrate_batchnorm_stats()
```

实现要求：

1. 不改变默认训练行为；
2. 默认 `adaptation.enabled: false`；
3. 只有配置显式打开时才启用；
4. AdaBN train-only 只使用 train fold；
5. AdaBN val-transductive 只使用 val fold 的 features；
6. 不使用 labels；
7. 不反向传播；
8. 只更新 BN running_mean / running_var；
9. 不更新模型权重；
10. 对每个 fold 单独执行，不能跨 fold 混用 BN 统计量。

建议配置格式：

```yaml
adaptation:
  enabled: false
  mode: none        # none / train_only / val_transductive
  reset_stats: true
  max_batches: null
  use_labels: false
```

---

### 3.2 实验 R1：原始路径 + AdaBN train-only

实验名：

```text
exp3_vote_5fold_adabn_train_only
```

输出目录：

```text
EEGNET/outputs_adabn_alignment/exp3_vote_5fold_adabn_train_only
```

设置：

```yaml
adaptation:
  enabled: true
  mode: train_only
  reset_stats: true
  use_labels: false
```

要求：

1. 使用与 R0 完全相同的 fold split。
2. 使用与 R0 完全相同的训练超参数。
3. 每个 fold 训练结束后，只在 train fold 上重校准 BN。
4. 然后在 val fold 上评估。
5. 不使用 val 数据做 adaptation。
6. setting = inductive。
7. 可以进入 inductive leaderboard。

---

### 3.3 实验 R2：原始路径 + AdaBN val-transductive

实验名：

```text
exp3_vote_5fold_adabn_val_transductive
```

输出目录：

```text
EEGNET/outputs_adabn_alignment/exp3_vote_5fold_adabn_val_transductive
```

设置：

```yaml
adaptation:
  enabled: true
  mode: val_transductive
  reset_stats: true
  use_labels: false
```

要求：

1. 使用与 R0 完全相同的 fold split。
2. 使用与 R0 完全相同的训练超参数。
3. 每个 fold 训练结束后，使用 val fold 的无标签 features 重校准 BN。
4. 然后在同一 val fold 上评估。
5. 不使用 val labels 进行 adaptation。
6. 不反向传播。
7. setting = transductive。
8. 不进入普通 inductive leaderboard。

---

## 4. 每个实验必须生成的文件

每个实验目录至少包含：

```text
summary_metrics.json
fold_metrics.csv
predictions.csv
classification_report.csv
confusion_matrix.png
roc_curve.png
pr_curve.png
run_metadata.json
config_snapshot.yaml
leakage_check.json
```

`leakage_check.json` 至少包含：

```json
{
  "experiment": "",
  "setting": "inductive/transductive",
  "uses_val_unlabeled_for_adaptation": false,
  "uses_val_labels_for_training": false,
  "uses_public_for_training": false,
  "uses_public_for_model_selection": false,
  "updates_bn_on_target": false,
  "updates_model_weights_on_target": false,
  "subject_overlap": false,
  "trial_overlap": false,
  "window_overlap": false,
  "normalization_fit_scope": "train_only",
  "risk_level": "low/medium/high/critical",
  "notes": ""
}
```

---

## 5. 判定规则

### 5.1 AdaBN 可以作为主线的条件

只有满足以下条件，才认为 AdaBN 对原始最优模型有真实收益：

1. R0 成功复现原始 `exp3_vote_5fold`；
2. R1 是 inductive setting；
3. R1 无 subject/trial/window 泄露；
4. R1 不使用 val/test/public 做 adaptation；
5. R1 相比 R0：
   - BA 提升 >= 0.005；
   - 或 Macro-F1 提升 >= 0.005；
6. R1 std 不明显增大；
7. 至少 3/5 fold 指标提升。

### 5.2 AdaBN 不进入主线的条件

若出现以下情况，则不纳入最终模型：

1. R0 无法复现原始 baseline；
2. R1 不超过 R0；
3. R1 提升 < 0.005；
4. R1 std 明显增大；
5. 只有 R2 transductive 有提升；
6. 结果依赖 val/test/public 无标签数据。

### 5.3 DANN 继续暂停的条件

本任务不实现 DANN。  
只有在 AdaBN 对齐完成并且 R1 确认有收益后，才考虑 DANN。

---

## 6. 最终输出文件

生成：

```text
EEGNET/outputs_adabn_alignment/alignment_report.md
EEGNET/outputs_adabn_alignment/config_diff_exp3_vs_adabn_off.md
EEGNET/outputs_adabn_alignment/codepath_diff_exp3_vs_adabn_off.md
EEGNET/outputs_adabn_alignment/split_diff_exp3_vs_adabn_off.csv
EEGNET/outputs_adabn_alignment/metric_scope_diff_exp3_vs_adabn_off.md
EEGNET/outputs_adabn_alignment/repro_vs_original.csv
EEGNET/outputs_adabn_alignment/adabn_on_original_eegnet_report.md
EEGNET/outputs_adabn_alignment/inductive_leaderboard.csv
EEGNET/outputs_adabn_alignment/transductive_leaderboard.csv
```

---

## 7. 最终报告必须回答的问题

在 `alignment_report.md` 中明确回答：

1. A control 为什么比原始 baseline 低 0.0287？
2. 主要原因是配置差异、代码路径差异、split 差异、评估口径差异，还是随机性？
3. 原始路径 R0 是否能复现 `exp3_vote_5fold`？
4. R0 与原始 `exp3_vote_5fold` 的 BA / Macro-F1 差异是多少？
5. 原始路径 R1 AdaBN train-only 是否超过 R0？
6. R1 的提升是否 >= 0.005？
7. R1 的 std 是否明显增大？
8. 原始路径 R2 AdaBN val-transductive 是否超过 R0？
9. R2 是否只作为 transductive 补充分析？
10. AdaBN 是否值得纳入最终主线？
11. DANN 是否继续推进？

---

## 8. 终端最终摘要格式

最后打印：

```text
===== AdaBN Alignment Summary =====

Original exp3_vote_5fold:
  BA:
  Macro-F1:
  ROC-AUC:

Fork A control:
  BA:
  Macro-F1:
  ROC-AUC:

Main cause of gap:
  config/code/split/metric/random/unknown

R0 original repro:
  BA:
  Macro-F1:
  ROC-AUC:
  reproduces original: yes/no

R1 original + AdaBN train-only:
  BA:
  Macro-F1:
  ROC-AUC:
  delta BA vs R0:
  delta Macro-F1 vs R0:
  improves >= 0.005: yes/no
  std increased: yes/no

R2 original + AdaBN val-transductive:
  BA:
  Macro-F1:
  ROC-AUC:
  setting: transductive

Leakage:
  subject overlap:
  trial overlap:
  window overlap:
  public used:
  high risk:
  critical risk:

Final recommendation:
===================================
```

---

## 9. 执行优先级

请严格按顺序执行：

1. 配置差异审计；
2. 代码路径差异审计；
3. split 差异审计；
4. 指标口径差异审计；
5. R0 原始路径复现；
6. R1 原始路径 AdaBN train-only；
7. R2 原始路径 AdaBN val-transductive；
8. 生成最终报告。

如果 R0 不能复现原始 `exp3_vote_5fold`，停止在第 5 步，不要继续跑 R1/R2。
