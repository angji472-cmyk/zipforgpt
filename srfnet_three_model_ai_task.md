# AI任务书：SRFNet 审计 + 三模型 Rank Ensemble + 最终候选决策

## 0. 当前背景

当前项目已经有三个强候选：

### 当前主提交

```text
Conformer + exp3 E3_rank ensemble
method = rank ensemble
w = 0.50
BA = 0.7917
MF1 = 0.7917
Decision = PRIMARY
```

已知 per-fold：

```text
fold0 = 0.7708
fold1 = 0.8958
fold2 = 0.8125
fold3 = 0.6875
fold4 = 0.7917
```

### 单模型候选 1：tuned Conformer

```text
eeg_conformer_lite_tuned
BA = 0.7750
MF1 = 0.7750
Decision = SUPPLEMENT
```

已知 per-fold：

```text
fold0 = 0.7500
fold1 = 0.8958
fold2 = 0.7292
fold3 = 0.7083
fold4 = 0.7917
```

### 单模型候选 2：exp3 baseline

```text
exp3_vote_alpha062_cwmean_top4 / exp3 related score
BA = 0.7583 frozen version
或 rebuild cwmean BA = 0.7458
Decision = BACKUP / score-source dependent
```

### 新模型候选：SRFNet

SRFNet 已实现为三支路门控融合架构：

```text
RawConformerBranch
+ FftEEGNetBranch
+ RegionalSummaryEncoder
+ Gated Fusion
+ Subject-level ranking loss
+ Trial consistency loss
```

目前已知 SRFNet 结果：

```text
srfnet_rank_consistency report: BA = 0.7583
auto-tune trial_0007: BA = 0.7667
trial0007_long_e6e10: BA = 0.7833  # 需要审计确认
```

当前最重要的问题：

```text
SRFNet trial0007_long_e6e10 是否真实、完整、可复现？
如果真实，能否和 Conformer + exp3 做三模型 rank ensemble，超过当前 0.7917？
```

---

## 1. 总目标

本任务分为四个核心部分：

1. 审计 SRFNet 最高分版本 `trial0007_long_e6e10`；
2. 审计 SRFNet 数据采样率和输入维度，排查 128Hz / 250Hz 描述不一致；
3. 将 SRFNet 加入现有 Conformer + exp3 ensemble，做三模型 rank ensemble；
4. 生成最终候选榜，决定是否替换当前主提交。

最终只允许三种结论：

```text
A. PROMOTE_THREE_MODEL_ENSEMBLE
B. KEEP_CURRENT_CONFORMER_EXP3_ENSEMBLE
C. KEEP_CURRENT_ENSEMBLE_AND_USE_SRFNET_AS_SUPPLEMENT
```

---

## 2. 最高优先级约束

### 2.1 禁止事项

以下行为禁止：

1. 禁止覆盖当前主提交：

```text
outputs/conformer_exp3_ensemble/public_submission_ensemble.xlsx
```

2. 禁止覆盖旧 frozen baseline：

```text
outputs/final_review/final_submission_clean.xlsx
```

3. 禁止覆盖 Conformer tuned 提交：

```text
outputs/eeg_conformer_tuning/final_candidate/public_submission_conformer_tuned.xlsx
```

4. 禁止使用 public test label。
5. 禁止使用 public 反馈调参。
6. 禁止 per-fold oracle 权重。
7. 禁止 per-subject 权重。
8. 禁止 per-subject 规则。
9. 禁止 stacking/meta-classifier。
10. 禁止重新训练 AdaBN / DANN / DENS / RobotFaces / GraphTransformer / ST-DADGAT-lite。
11. 禁止把没有完整 5-fold 的结果写成最终结论。
12. 禁止把单 fold 提升写成主模型提升。
13. 禁止删除、移动、覆盖旧实验结果。
14. 禁止为了追求分数修改 top-4 规则。
15. 禁止 public submission 参与模型选择。

### 2.2 允许事项

允许：

1. 新建输出目录；
2. 读取已有 SRFNet / Conformer / exp3 score 文件；
3. 从 predictions/top4 CSV 复算指标；
4. 重建缺失的 original_trial score；
5. 做 score-level / rank-level ensemble；
6. 必要时复跑 SRFNet trial0007_long 同配置以确认复现；
7. 生成新候选提交文件；
8. 做提交审计；
9. 做 SHA256 hash 冻结；
10. 输出最终报告。

---

## 3. 输出总目录

所有新结果写入：

```text
outputs/srfnet_three_model_final/
```

建议目录结构：

```text
outputs/srfnet_three_model_final/
├── 00_source_inventory/
├── 01_srfnet_long_audit/
├── 02_sampling_rate_audit/
├── 03_score_alignment/
├── 04_three_model_ensemble/
├── 05_submission_candidates/
├── 06_final_audit/
├── 07_reports/
└── run_state/
```

每一步更新：

```text
outputs/srfnet_three_model_final/run_state/progress.md
```

---

## 4. Phase 0：Source Inventory

## 4.1 目标

找到所有相关输入文件，明确哪些结果真实存在，哪些只是报告中提到。

## 4.2 搜索对象

必须搜索以下路径：

```text
outputs/srfnet/
auto_tune_runs/srfnet_first_live/
outputs/conformer_exp3_ensemble/
outputs/eeg_conformer_tuning/
outputs/final_review/
outputs/exp3_aggregation_top4_sweep/
outputs/final_7h_tuning_run/
```

如果实际路径不同，请自动全项目搜索：

```text
srfnet
trial0007
trial_0007
long_e6e10
conformer_exp3
E3_rank
rank_ensemble
original_trial_scores
top4_predictions_cv
fold_metrics
summary_metrics
public_submission
```

## 4.3 必须登记的文件

生成：

```text
outputs/srfnet_three_model_final/00_source_inventory/source_inventory.csv
```

字段：

```csv
name,path,exists,file_type,model,version,contains_cv_scores,contains_public_scores,contains_submission,notes
```

必须包含：

1. SRFNet model file；
2. SRFNet train script；
3. SRFNet trial0007 config；
4. SRFNet trial0007_long result；
5. SRFNet top4 predictions；
6. SRFNet original_trial_scores；
7. Conformer tuned original_trial_scores；
8. exp3 score source；
9. current ensemble score source；
10. current ensemble submission；
11. frozen baseline submission。

## 4.4 输出

```text
outputs/srfnet_three_model_final/00_source_inventory/source_inventory.csv
outputs/srfnet_three_model_final/00_source_inventory/source_inventory_report.md
```

---

## 5. Phase 1：SRFNet trial0007_long_e6e10 审计

## 5.1 目标

确认 SRFNet 最高分版本：

```text
trial0007_long_e6e10
BA = 0.7833
```

是否真实、完整、可复现。

## 5.2 必须检查

1. 是否存在完整结果目录；
2. 是否有 5 个 fold；
3. 是否每个 fold 都有 checkpoint / prediction / metric；
4. 是否有 `top4_predictions_cv.csv`；
5. 是否有 `original_trial_scores.csv`；
6. 是否有 `fold_metrics.csv`；
7. 是否有 `summary_metrics.json`；
8. 是否有 public submission；
9. 是否使用与 exp3 / Conformer 相同 subject split；
10. 是否没有 public label；
11. 是否没有 validation/test 泄露；
12. 是否没有 per-fold oracle；
13. 是否没有 per-subject tuning；
14. 是否 top-4 口径一致。

## 5.3 指标复算

从 `top4_predictions_cv.csv` 或 fold-level top4 predictions 重新计算：

```text
Balanced Accuracy
Macro-F1
Accuracy
Per-fold BA
Confusion matrix
Per-subject accuracy
```

期望结果：

```text
fold0 = 0.7500
fold1 = 0.8333
fold2 = 0.7917
fold3 = 0.7917
fold4 = 0.7500
mean BA = 0.7833
```

如果实际文件不匹配，以文件复算结果为准，并在报告中明确说明。

## 5.4 判定

### SUPPLEMENT_STRONG

条件：

```text
BA >= 0.7800
完整 5-fold
可复算
无泄露
```

### SUPPLEMENT_WEAK

条件：

```text
0.7583 <= BA < 0.7800
完整 5-fold
无泄露
```

### NEEDS_REBUILD

条件：

```text
报告写了 0.7833，但缺少可复算文件
或文件不完整
```

### INVALID

条件：

```text
public label 使用
subject overlap
trial/window leakage
per-fold oracle
```

## 5.5 输出

```text
outputs/srfnet_three_model_final/01_srfnet_long_audit/
├── srfnet_long_metric_recalculation.csv
├── srfnet_long_metric_recalculation.md
├── srfnet_long_split_audit.md
├── srfnet_long_prediction_integrity.csv
├── srfnet_long_public_submission_audit.md
└── srfnet_long_final_decision.md
```

---

## 6. Phase 2：Sampling Rate Audit

## 6.1 背景

赛题说明中训练集和测试集采样率是：

```text
250Hz
```

但某些 SRFNet 资料中写了：

```text
128Hz
```

必须确认这是 typo 还是实际处理错误。

## 6.2 检查内容

1. 原始 `.mat` 数据实际采样率；
2. manifest 中每个样本的 `n_times`；
3. SRFNet 输入 `n_times`；
4. `RawConformerBranch` 初始化时使用的 `n_times`；
5. 是否有 resample 代码；
6. 若有 resample，目标采样率是多少；
7. patch_size=32 对应多少秒；
8. report 中 128Hz 是否只是旧文字；
9. 训练实际是否使用了 250Hz 下的 10s windows，即 2500 samples；
10. public inference 是否与训练一致。

## 6.3 判定

### PASS

```text
实际训练/推理均使用 250Hz 或正确重采样，报告 128Hz 只是 typo
```

### WARNING

```text
存在重采样，但训练和推理一致，且记录清楚
```

### FAIL

```text
训练/推理采样率不一致
或把 250Hz 错当 128Hz 解释，导致窗口/patch 错误
```

## 6.4 输出

```text
outputs/srfnet_three_model_final/02_sampling_rate_audit/
├── sampling_rate_audit.md
├── n_times_check.csv
└── preprocessing_path_check.md
```

---

## 7. Phase 3：Score Alignment

## 7.1 目标

把三类模型的 original_trial score 对齐到同一 key：

```text
fold
subject_id
original_trial_id
y_true
score
```

模型包括：

1. Conformer tuned；
2. exp3 selected source；
3. SRFNet trial0007_long；
4. current Conformer+exp3 ensemble。

## 7.2 必须处理的 score source

### Conformer

寻找：

```text
outputs/eeg_conformer_tuning/final_candidate/original_trial_scores.csv
```

或等价文件。

### exp3

优先使用 Team A 已确认来源：

```text
exp3 rebuild cwmean BA=0.7458
```

同时如果 frozen exp3 0.7583 score 存在，也一并纳入：

```text
exp3_frozen
exp3_rebuild
```

### SRFNet

优先使用：

```text
trial0007_long_e6e10 original_trial_scores.csv
```

如果不存在，尝试从 fold-level predictions 汇总。

## 7.3 对齐检查

必须检查：

1. 三个模型是否覆盖相同 fold；
2. 每 fold 是否覆盖相同 subject；
3. 每 subject 是否有 8 个 original_trial；
4. `y_true` 是否一致；
5. original_trial_id 是否一致；
6. 是否存在重复 key；
7. 是否存在缺失 key；
8. 是否每个模型 top-4 后能复现各自报告 BA。

## 7.4 输出

```text
outputs/srfnet_three_model_final/03_score_alignment/
├── aligned_scores_long.csv
├── score_alignment_report.md
├── missing_or_duplicate_keys.csv
├── model_score_reproduction.csv
└── per_model_top4_recomputed.csv
```

---

## 8. Phase 4：Three-Model Rank Ensemble

## 8.1 目标

尝试三模型 rank ensemble：

```text
Conformer tuned
+ exp3
+ SRFNet
```

目标超过当前 best：

```text
Conformer + exp3 E3_rank w=0.50
BA = 0.7917
```

尤其要修复当前 best 的 fold3：

```text
current best fold3 = 0.6875
```

如果 SRFNet long 的 fold3=0.7917 可复现，它可能帮助修复 fold3。

## 8.2 统一规则

所有 ensemble 都在 original_trial 层面做：

```text
每个 subject 内 8 个 trial 排序
top 4 → 1
bottom 4 → 0
```

禁止：

```text
threshold
per-fold weight
per-subject weight
public feedback
stacking
```

---

## 8.3 Ensemble 方法

### E1：三模型 equal rank average

```text
rank = mean(rank_conformer, rank_exp3, rank_srfnet)
```

### E2：三模型 weighted rank grid

权重候选：

```text
w_conformer ∈ {0.20, 0.25, 0.30, 0.33, 0.40, 0.45, 0.50}
w_exp3      ∈ {0.15, 0.20, 0.25, 0.30, 0.33, 0.40}
w_srfnet    ∈ {0.15, 0.20, 0.25, 0.30, 0.33, 0.40}
```

只保留：

```text
w_conformer + w_exp3 + w_srfnet = 1.0 ± 1e-6
```

### E3：三模型 reciprocal rank

```text
score = wc / (rank_conformer + eps)
      + we / (rank_exp3 + eps)
      + ws / (rank_srfnet + eps)
```

同样使用固定全局权重。

### E4：三模型 percentile rank

每个 subject 内：

```text
percentile_rank = rank / 8
```

然后加权平均。

### E5：Leave-one-model-out

比较：

```text
Conformer + exp3
Conformer + SRFNet
exp3 + SRFNet
Conformer + exp3 + SRFNet
```

### E6：Stable objective

不要只按 mean BA 排序，同时计算：

```text
stable_score = BA_mean - 0.25 * BA_std
min_fold_BA
fold3_BA
folds_ge_current_best
folds_ge_conformer
folds_ge_srfnet
```

---

## 8.4 Promote 标准

新 ensemble 只有满足以下条件才替代当前 best：

```text
BA > 0.7917
MF1 >= 0.7917
至少 3/5 folds >= current best 对应 fold
std 不明显增加
public 未参与选择
全局固定方法和权重
```

如果：

```text
BA = 0.7917
但 fold3 明显改善
且 std 更低
```

则标记：

```text
STABILITY_PROMOTE_CANDIDATE
```

需要人工确认是否替代。

否则：

```text
KEEP_CURRENT_BEST
```

## 8.5 输出

```text
outputs/srfnet_three_model_final/04_three_model_ensemble/
├── three_model_rank_leaderboard.csv
├── three_model_per_fold_results.csv
├── leave_one_model_out.csv
├── stable_objective_leaderboard.csv
├── fold3_repair_report.md
├── best_three_model_top4_predictions.csv
├── best_three_model_original_trial_scores.csv
├── public_submission_best_three_model.xlsx
├── public_submission_best_three_model_with_prob.xlsx
└── three_model_ensemble_report.md
```

---

## 9. Phase 5：Public Submission Candidate Generation

## 9.1 候选

至少准备以下候选：

```text
C0: current Conformer+exp3 E3_rank w=0.50
C1: tuned Conformer
C2: exp3 frozen baseline
C3: SRFNet trial0007_long
C4: best three-model ensemble
```

## 9.2 生成规则

所有 public submission 必须：

```text
每 user 8 trials
top 4 positive
bottom 4 neutral
```

## 9.3 审计

每个 submission 检查：

```text
80 rows
10 users
8 trials/user
4 positive/user
4 neutral/user
no duplicated user_id+trial_id
Emotion_label ∈ {0,1}
columns match template
Excel readable
```

## 9.4 输出

```text
outputs/srfnet_three_model_final/05_submission_candidates/
├── submission_candidate_list.csv
├── submission_audit_all.md
├── C0_current_ensemble.xlsx
├── C1_conformer_tuned.xlsx
├── C2_exp3_frozen.xlsx
├── C3_srfnet_long.xlsx
├── C4_best_three_model.xlsx
└── hashes.txt
```

---

## 10. Phase 6：Final Candidate Board

## 10.1 汇总候选

生成：

```text
outputs/srfnet_three_model_final/06_final_audit/final_candidate_board.csv
```

字段：

```csv
candidate,model_family,ba,mf1,ba_std,fold0,fold1,fold2,fold3,fold4,folds_ge_current_best,min_fold_ba,submission_path,sha256,decision,reason
```

## 10.2 决策规则

### PRIMARY

```text
BA 最高
且 BA > 0.7917
且 folds_ge_current_best >= 3
且 submission audit PASS
```

### PRIMARY_STABILITY

```text
BA = 0.7917
但 fold3 / min_fold / std 明显优于 current best
```

### SUPPLEMENT_STRONG

```text
0.7833 <= BA < 0.7917
或三模型 ensemble 未提升但修复 fold3
```

### BACKUP

```text
旧 baseline / Conformer / SRFNet 单模型
```

### REJECT

```text
低于 0.7750
或不完整
或无法复现
```

## 10.3 输出

```text
outputs/srfnet_three_model_final/06_final_audit/
├── final_candidate_board.csv
├── final_decision.md
├── final_submission_selected.xlsx
├── final_submission_selected_with_prob.xlsx
├── final_submission_audit.md
└── final_hashes.txt
```

---

## 11. Phase 7：最终报告

生成：

```text
outputs/srfnet_three_model_final/07_reports/
├── srfnet_three_model_final_report.md
├── final_recommendation.md
├── team_update.md
└── technical_summary_for_ppt.md
```

## 11.1 final_report 必须包含

```markdown
# SRFNet + Three-Model Ensemble Final Report

## 1. Motivation

为什么加入 SRFNet：
- 当前 ensemble fold3 弱
- SRFNet long 可能 fold2/fold3 强
- 三模型 rank ensemble 可能修复负互补

## 2. SRFNet Architecture

描述：
- RawConformerBranch
- FftEEGNetBranch
- RegionalSummaryEncoder
- gated fusion
- ranking loss
- consistency loss

## 3. SRFNet Audit

说明：
- trial0007_long 是否真实
- BA 是否可复现
- split/leakage 结果
- sampling rate audit

## 4. Score Alignment

说明：
- Conformer score source
- exp3 score source
- SRFNet score source
- key alignment 是否通过

## 5. Three-Model Ensemble

展示：
- best method
- weights
- BA/MF1
- per-fold
- fold3 是否修复

## 6. Final Candidate Board

展示所有候选。

## 7. Final Recommendation

明确：
- 主提交
- 备份提交
- 是否替换 current best

## 8. Caveats

包括：
- CV model selection bias
- public/private 可能不同
- top-4 domain prior
- SRFNet long 若不可复现则不能主用
```

## 11.2 team_update.md

写给队友：

```markdown
# 队内更新：SRFNet 与三模型 ensemble

我们审计了 SRFNet 最高分版本，并尝试将它加入当前 Conformer+exp3 ensemble。

当前最终推荐：
...

为什么：
...

是否超过 0.7917：
...

提交文件：
...
```

## 11.3 technical_summary_for_ppt.md

用于答辩：

```markdown
# 技术亮点

1. Subject-level top-4 结构化解码
2. Conformer 捕捉时序局部-全局特征
3. FFT/EEGNet 捕捉频域节律特征
4. Rank ensemble 利用模型互补排序
5. SRFNet 尝试将 rank loss 内化到端到端模型中
```

---

## 12. 最终终端摘要格式

任务结束后打印：

```text
===== SRFNet Three-Model Final Summary =====

Current best before SRFNet:
  Conformer + exp3 E3_rank w=0.50
  BA:
  MF1:

SRFNet audit:
  trial0007_long exists:
  metric reproducible:
  BA:
  decision:

Sampling rate audit:
  result:
  actual n_times:
  resampling:
  decision:

Score alignment:
  Conformer:
  exp3:
  SRFNet:
  alignment:

Three-model ensemble:
  best method:
  weights:
  BA:
  MF1:
  fold0:
  fold1:
  fold2:
  fold3:
  fold4:
  decision:

Final selected:
  model:
  submission:
  SHA256:

Recommendation:
  PROMOTE_THREE_MODEL / KEEP_CURRENT_ENSEMBLE / KEEP_CURRENT_WITH_SRFNET_SUPPLEMENT

===========================================
```

---

## 13. 最终执行原则

这次任务的核心不是继续盲目调参，而是回答：

```text
SRFNet 是否能作为第三个互补模型，
把当前 Conformer+exp3 rank ensemble 从 0.7917 继续推高？
```

如果三模型 ensemble 没有超过当前 best：

```text
保持 Conformer + exp3 E3_rank w=0.50 为主提交。
```

如果 SRFNet long 可复现但没超过：

```text
把 SRFNet 作为强补充和答辩创新点。
```

如果三模型 ensemble 超过：

```text
冻结三模型 ensemble 为新主提交。
```
