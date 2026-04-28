# EEG-Conformer-lite 定向调参计划：只围绕当前最接近挑战者继续优化

## 0. 当前背景

当前安全主模型：

```text
exp3_vote_alpha062_cwmean_top4
BA = 0.7583
MF1 = 0.7583
```

当前最接近的新模型：

```text
eeg_conformer_lite
Full 5-fold BA = 0.7500
vs final = -0.0083
```

已知 per-fold 表现：

```text
fold_0: 0.7708 vs final 0.7292  => +0.0416
fold_1: 0.8542 vs final 0.8542  =>  0.0000
fold_2: 0.6875 vs final 0.7708  => -0.0833
fold_3: 0.7500 vs final 0.7500  =>  0.0000
fold_4: 0.6875 vs final 0.6875  =>  0.0000
```

关键结论：

```text
eeg_conformer_lite 不是整体失败，而是 fold_2 明显拖后腿。
```

因此本次调参目标不是乱调大网格，而是：

1. 修复 fold_2 崩溃；
2. 保持 fold_0 的优势；
3. 不牺牲 fold_1 / fold_3 / fold_4；
4. 如果成功，再做完整 5-fold；
5. 如果失败，停止 Conformer，不再投入。

---

## 1. 总目标

目标模型：

```text
eeg_conformer_lite_tuned
```

目标超过：

```text
exp3_vote_alpha062_cwmean_top4
BA = 0.7583
```

成功标准：

```text
Full 5-fold BA > 0.7583
MF1 >= 0.7583
至少 3/5 folds >= final 对应 fold
std 不明显增大
无 public test 调参
无 per-fold oracle
```

最低可接受为 supplement：

```text
BA >= 0.7500
且 fold_2 明显改善
但未超过 final
```

否则 reject。

---

## 2. 最高优先级约束

### 2.1 禁止事项

1. 禁止覆盖 frozen final：

```text
outputs/final_review/final_submission_clean.xlsx
```

2. 禁止覆盖原 `eeg_conformer_lite` 结果。
3. 禁止使用 public test label。
4. 禁止用 public 反馈调参。
5. 禁止 per-fold oracle alpha / threshold。
6. 禁止 stacking/meta-classifier。
7. 禁止引入外部数据。
8. 禁止重开 AdaBN / DANN / GraphTransformer / DENS / RobotFaces。
9. 禁止一上来跑 full 大网格。
10. 禁止只看 fold_1 高分就 promote。
11. 禁止未完成 5-fold 就写“超过 baseline”。
12. 禁止只改后处理而不记录。
13. 禁止训练时使用 validation/test 统计量。

### 2.2 必须遵守

1. 使用与 final 完全相同 subject split。
2. 使用 original_trial + subject top-4 统一口径。
3. public 只用于最终 inference，不参与任何选择。
4. 所有输出放到：

```text
outputs/eeg_conformer_tuning/
```

5. 所有实验必须记录：
   - config
   - seed
   - fold
   - metrics
   - checkpoint
   - predictions
   - postprocess
   - decision
6. 每轮调参必须先跑 screening folds。
7. 只有通过 screening 才跑完整 5-fold。

---

## 3. 输出目录

```text
outputs/eeg_conformer_tuning/
├── 00_reference/
├── 01_diagnostics/
├── 02_screening/
│   ├── depth_width/
│   ├── dropout/
│   ├── lr_wd/
│   ├── patch_kernel/
│   ├── pooling/
│   └── regularization/
├── 03_full_5fold/
├── 04_postprocess/
├── 05_ensemble/
├── 06_submission/
├── 07_reports/
└── run_state/
```

进度日志：

```text
outputs/eeg_conformer_tuning/run_state/progress.md
```

---

## 4. Phase 0：Reference 固化

先读取并固化现有结果，不要重新解释。

生成：

```text
outputs/eeg_conformer_tuning/00_reference/reference_metrics.md
outputs/eeg_conformer_tuning/00_reference/reference_per_fold.csv
```

必须包含：

```text
final baseline:
exp3_vote_alpha062_cwmean_top4
BA = 0.7583

current conformer:
eeg_conformer_lite
BA = 0.7500
```

记录当前 conformer 配置：

```text
emb_size
depth
num_heads
dropout
patch_size / kernel size
batch_size
lr
weight_decay
epochs
patience
optimizer
scheduler
```

如果配置文件缺失，从训练脚本和日志中恢复。

---

## 5. Phase 1：Fold 2 诊断

## 5.1 目标

确定 fold_2 为什么拖后腿：

```text
final fold_2 BA = 0.7708
conformer fold_2 BA = 0.6875
delta = -0.0833
```

## 5.2 必须检查

1. fold_2 test subjects 是谁？
2. fold_2 中哪些 subject 错误最多？
3. final 模型和 conformer 在 fold_2 错误是否重叠？
4. conformer 是排序失败，还是概率校准失败？
5. conformer 在 fold_2 的 score range 是否过窄？
6. fold_2 是否存在 DEP / HC 特定失败？
7. fold_2 是否是某些 original_trial 排序被 top-4 反转？
8. conformer 是否在 fold_0 高、fold_2 低，说明过拟合某些 subject 类型？

## 5.3 输出

```text
outputs/eeg_conformer_tuning/01_diagnostics/fold2_diagnostics.md
outputs/eeg_conformer_tuning/01_diagnostics/fold2_error_subjects.csv
outputs/eeg_conformer_tuning/01_diagnostics/final_vs_conformer_fold2_trial_comparison.csv
```

## 5.4 诊断结论类型

诊断结论只能是：

```text
calibration_issue
overfitting_issue
underfitting_issue
subject_shift_issue
ranking_issue
unknown
```

后续调参要根据诊断走。

---

## 6. Phase 2：筛选 folds 设计

所有调参先跑 screening，不直接 full 5-fold。

### 6.1 默认 screening folds

```text
fold_2: 主要短板 fold
fold_0: conformer 当前优势 fold
fold_4: 难 fold，防过拟合
```

如果时间紧，至少跑：

```text
fold_2
fold_4
```

### 6.2 Screening 通过规则

一个配置进入完整 5-fold 必须满足：

```text
fold_2 BA >= 0.7292
```

并且：

```text
fold_4 BA >= 0.6875
```

并且满足以下之一：

```text
screen_avg_BA >= current_conformer_screen_avg
或 fold_2 提升 >= +0.0417
```

强通过：

```text
fold_2 BA >= 0.7500
且 fold_4 >= 0.6875
```

直接拒绝：

```text
fold_2 <= 0.6875 且 fold_4 无提升
或 OOM 后降级仍失败
或训练明显不稳定
```

---

## 7. Phase 3：调参顺序

严格按下面顺序执行。  
不要跳到大网格。

---

# T1：正则化优先调参

因为 fold_2 明显掉点，优先怀疑过拟合或泛化不足。

## T1.1 Dropout sweep

当前 dropout 记为 `d0`。  
只试：

```text
d0 + 0.1
d0 + 0.2
d0 - 0.1
```

如果当前 d0 未知，使用：

```text
0.3
0.5
0.6
```

推荐：

```yaml
dropout_candidates: [0.3, 0.5, 0.6]
```

先跑：

```text
fold_2
fold_4
```

输出：

```text
outputs/eeg_conformer_tuning/02_screening/dropout/dropout_screening.csv
outputs/eeg_conformer_tuning/02_screening/dropout/dropout_screening_report.md
```

晋级条件：

```text
fold_2 >= 0.7292
且 fold_4 >= 0.6875
```

---

## T1.2 Weight decay sweep

只试：

```text
weight_decay = [0.005, 0.01, 0.02, 0.05]
```

保持其它参数为当前最佳 dropout。

目标：

```text
让 fold_2 提升，不牺牲 fold_4
```

---

## T1.3 Label smoothing

只试：

```text
label_smoothing = [0.0, 0.05, 0.1]
```

注意：

- 只在 CE loss 中加入；
- 不能影响 top-4；
- 如果造成概率过平，可能伤害排序，要记录 score separation。

---

# T2：模型容量调参

如果 T1 无效，再调容量。

## T2.1 emb_size / depth

当前可能为：

```text
emb_size=32
depth=2
heads=4
```

只试：

```text
A: emb_size=24, depth=2, heads=4
B: emb_size=32, depth=1, heads=4
C: emb_size=48, depth=2, heads=4
D: emb_size=32, depth=3, heads=4
```

8GB OOM 时跳过 D。

预期：

```text
如果过拟合：A/B 可能改善 fold_2
如果欠拟合：C/D 可能改善
```

筛选 folds：

```text
fold_2
fold_4
```

---

## T2.2 attention heads

只试：

```text
heads = [2, 4, 8]
```

前提：

```text
emb_size % heads == 0
```

如果 emb_size=32，则：

```text
heads 2,4,8 均可
```

目标：

```text
检查 attention 粒度是否影响 ranking
```

---

# T3：Patch / convolution front-end 调参

Conformer 对前端 patch/conv 很敏感。只做小范围。

## T3.1 Temporal kernel / patch size

只试：

```text
patch_kernel = [16, 32, 64]
```

或如果以时间秒定义：

```text
0.25s, 0.5s, 1.0s
```

必须与采样率 250Hz 对齐。

目的：

```text
小 kernel 捕捉短期节律
大 kernel 捕捉更稳的情绪段模式
```

## T3.2 Stride / pooling

只试：

```text
stride = [4, 8, 16]
```

避免 token 太多 OOM。

---

# T4：训练策略调参

## T4.1 Learning rate

只试：

```text
lr = [0.0002, 0.0005, 0.0008]
```

不要超过 0.001。

## T4.2 Scheduler

只试：

```text
cosine
plateau
onecycle
```

如果 onecycle 不稳定，立即停止。

## T4.3 Patience

只试：

```text
patience = [10, 15, 20]
```

如果训练很慢，保留 12 或 15。

---

# T5：Conformer 后处理专属调参

如果某个 conformer 配置接近 final，才做后处理。

## T5.1 Aggregation

比较：

```text
mean_prob
confidence_weighted_mean
mean_logit
rank_average
median_prob
```

## T5.2 Top-4 保持固定

不能取消 top-4。

## T5.3 Calibration

只允许全局温度：

```text
temperature = [0.8, 1.0, 1.2, 1.5]
```

如果只有 probabilities，没有 logits，则跳过。

---

## 8. Full 5-fold 晋级

筛选后选择最多 2 个配置进入 full 5-fold：

```text
best_regularized_config
best_capacity_config
```

不要超过 2 个，防止时间爆炸。

完整 5-fold 输出：

```text
outputs/eeg_conformer_tuning/03_full_5fold/{config_name}/
├── fold_0/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
├── branch_or_prob_scores.csv
├── original_trial_scores.csv
├── top4_predictions_cv.csv
├── fold_metrics.csv
├── summary_metrics.json
└── full5_report.md
```

---

## 9. Conformer + Final 小权重 ensemble

如果某个 Conformer 配置 BA >= 0.7500，允许做小权重 ensemble。

## 9.1 权重

只试：

```text
final_weight = [0.8, 0.85, 0.9, 0.95]
conformer_weight = 1 - final_weight
```

最终 score：

```text
score = final_weight * final_score + conformer_weight * conformer_score
```

然后 subject top-4。

## 9.2 禁止

禁止：

```text
per-fold ensemble weight
per-subject ensemble weight
stacking
public selection
```

## 9.3 Promote 条件

```text
BA > 0.7583
至少 3/5 folds >= final
std 不明显增加
```

输出：

```text
outputs/eeg_conformer_tuning/05_ensemble/conformer_final_ensemble.csv
outputs/eeg_conformer_tuning/05_ensemble/conformer_final_ensemble_report.md
```

---

## 10. 8GB 显存安全配置

默认：

```yaml
batch_size: 8
mixed_precision: true
num_workers: 2
pin_memory: true
persistent_workers: false
gradient_clip_norm: 1.0
```

OOM 自动：

```text
batch_size 8 -> 4
emb_size halve if possible
depth reduce to 1
retry once
```

每 fold 后：

```python
del model, optimizer, dataloader
torch.cuda.empty_cache()
gc.collect()
```

如果某配置 OOM 两次：

```text
mark reject_oom
continue next config
```

不要卡住。

---

## 11. 最终候选选择

生成：

```text
outputs/eeg_conformer_tuning/06_submission/conformer_candidate_summary.csv
outputs/eeg_conformer_tuning/06_submission/conformer_candidate_decision.md
```

字段：

```csv
candidate,ba,mf1,std,fold0,fold1,fold2,fold3,fold4,folds_ge_final,decision,reason
```

决策：

### promote

```text
BA > 0.7583
MF1 >= 0.7583
folds_ge_final >= 3
std not worse
no leakage
```

### supplement

```text
0.7500 <= BA <= 0.7583
or fold_2 fixed but overall not higher
```

### reject

```text
BA < 0.7500
or fold_2 still <= 0.6875
or unstable
```

---

## 12. 最终提交文件

如果 promote，生成：

```text
outputs/eeg_conformer_tuning/06_submission/public_submission_conformer_tuned.xlsx
outputs/eeg_conformer_tuning/06_submission/public_submission_conformer_tuned_with_prob.xlsx
```

如果不 promote，不生成 final selected，只生成 candidate。

必须审计：

```text
80 rows
10 users
8 trials/user
4 positive/user
4 neutral/user
no duplicate user_id+trial_id
Emotion_label ∈ {0,1}
```

---

## 13. 报告

生成：

```text
outputs/eeg_conformer_tuning/07_reports/conformer_tuning_report.md
outputs/eeg_conformer_tuning/07_reports/conformer_tuning_leaderboard.csv
outputs/eeg_conformer_tuning/07_reports/team_update_conformer.md
```

报告必须包含：

1. 为什么调 Conformer；
2. 原始 Conformer 的 per-fold 问题；
3. fold_2 诊断；
4. 每轮筛选结果；
5. full 5-fold 结果；
6. 是否超过 final；
7. 是否推荐继续；
8. 如果失败，明确停止条件；
9. 如果接近，说明作为 supplement 的价值。

---

## 14. 终端摘要

任务完成后打印：

```text
===== EEG-Conformer Tuning Summary =====

Reference final:
  BA:
  MF1:

Original conformer:
  BA:
  fold2:
  decision:

Best screening config:
  name:
  fold2:
  fold4:

Full 5-fold candidates:
  candidate 1:
    BA:
    MF1:
    folds_ge_final:
  candidate 2:
    BA:
    MF1:
    folds_ge_final:

Best conformer:
  name:
  BA:
  MF1:
  vs final:
  decision:

Ensemble with final:
  best_weight:
  BA:
  decision:

Recommendation:
  PROMOTE / SUPPLEMENT / REJECT

=================================
```

---

## 15. 执行原则

这次调参只回答一个问题：

```text
EEG-Conformer-lite 是否能通过合理正则化和容量调整，稳定超过 exp3_vote_alpha062_cwmean_top4？
```

如果答案是否定的，就停止 Conformer。  
不要把 Conformer 扩展成无休止的大网格。
