# Agent Teams 总任务书：Conformer + exp3 Ensemble 深挖与最终候选决策

## 0. 当前项目状态

当前最强候选：

```text
Conformer + exp3 rank ensemble
method = E3_rank
w = 0.50
BA = 0.7917
MF1 = 0.7917
```

当前单模型候选：

```text
eeg_conformer_lite_tuned
BA = 0.7750
MF1 = 0.7750
```

旧安全基线：

```text
exp3_vote_alpha062_cwmean_top4
BA = 0.7583
MF1 = 0.7583
```

当前 ensemble 结果：

```text
Conformer:
fold0=0.7500
fold1=0.8958
fold2=0.7292
fold3=0.7083
fold4=0.7917

exp3:
fold0≈0.7292
fold1≈0.7917 或 0.8542，取决于 score 来源
fold2≈0.7708
fold3≈0.7500
fold4≈0.6875

Current best ensemble:
E3_rank w=0.50
fold0=0.7708
fold1=0.8958
fold2=0.8125
fold3=0.6875
fold4=0.7917
BA=0.7917
```

关键观察：

1. Conformer 在 fold0/fold1/fold4 强。
2. exp3 在 fold2/fold3 更稳。
3. rank ensemble 修复了 fold2，但 fold3 被拉低。
4. 现在最有价值的方向不是新大模型，而是：
   - score lineage 审计；
   - rank ensemble 细扫；
   - fold3 负互补诊断；
   - tuned Conformer 多 seed；
   - Conformer fold2/fold3 小范围调参；
   - 最终候选榜与提交审计。

---

## 1. 最高优先级约束

### 1.1 禁止事项

1. 禁止覆盖以下安全提交：

```text
outputs/final_review/final_submission_clean.xlsx
```

2. 禁止覆盖 Conformer 已生成提交：

```text
outputs/eeg_conformer_tuning/final_candidate/public_submission_conformer_tuned.xlsx
```

3. 禁止覆盖当前 ensemble 提交：

```text
outputs/conformer_exp3_ensemble/public_submission_ensemble.xlsx
```

4. 禁止使用 public test label。
5. 禁止使用 public 反馈进行调参。
6. 禁止 per-fold oracle 权重。
7. 禁止 per-subject 权重。
8. 禁止 stacking/meta-classifier。
9. 禁止重新训练旧失败路线：
   - AdaBN
   - DANN
   - DENS
   - RobotFaces
   - GraphTransformer
   - ST-DADGAT-lite
   - 大规模 SupCon
   - MMD
10. 禁止把单 fold 提升写成最终提升。
11. 禁止未完成 5-fold 就写 promote。
12. 禁止删除旧结果。
13. 禁止移动旧结果。
14. 禁止用 public submission 文件反推调参。

### 1.2 允许事项

允许：

1. 新建输出目录。
2. 读取已有 score/prob/rank 文件。
3. 重算 CV 指标。
4. 做 score-level/rank-level ensemble。
5. 训练 tuned Conformer 多 seed。
6. 做 Conformer 小范围超参调优。
7. 生成多个候选提交文件。
8. 做提交审计。
9. 做 hash 冻结。
10. 生成最终 candidate leaderboard。

---

## 2. 统一评估口径

所有候选必须统一使用：

```text
original_trial level
subject-level top-4
每个 subject 8 个 original_trial
ensemble score / model score 最高 4 个 → Emotion_label=1
其余 4 个 → Emotion_label=0
```

所有 CV 指标：

```text
Balanced Accuracy
Macro-F1
Accuracy
Per-fold BA
Per-subject accuracy
Confusion matrix
```

所有候选必须与当前 best 比较：

```text
Current best:
Conformer + exp3 E3_rank w=0.50
BA = 0.7917
```

Promote 条件：

```text
BA > 0.7917
MF1 >= 0.7917
至少 3/5 folds >= current best 对应 fold
std 不明显增加
无泄露
public 未参与调参
使用全局固定方法/权重
```

如果未达到：

```text
只作为 supplement，不替代 current best。
```

---

# 3. Agent Team A — Score Lineage & Audit

## 3.1 目标

确认当前 ensemble 使用的 score 来源，尤其是 exp3 到底来自：

```text
frozen exp3 BA=0.7583
```

还是：

```text
deterministic rebuild exp3 BA≈0.7458
```

当前 ensemble 报告中 exp3 显示为 0.7458，因此必须查清楚来源。

## 3.2 必须回答

1. Conformer original_trial scores 来源文件是什么？
2. exp3 original_trial scores 来源文件是什么？
3. exp3 score 对应的 CV BA 是 0.7458 还是 0.7583？
4. frozen exp3 的 original_trial scores 是否还存在？
5. deterministic rebuild exp3 的 original_trial scores 是否存在？
6. public ensemble 用的是哪一套 exp3 score？
7. 如果 frozen exp3 score 不存在，是否能从权威 CSV 或旧缓存恢复？
8. 如果不能恢复，报告中必须说明。

## 3.3 必须重跑的对比

如果找到多套 exp3 score，请重跑：

```text
A1: Conformer + exp3_rebuild rank ensemble
A2: Conformer + exp3_frozen rank ensemble
A3: Conformer + exp3_meanprob rank ensemble
A4: Conformer + exp3_cwmean rank ensemble
```

统一：

```text
method = rank ensemble
w grid = 0.00 to 1.00, step=0.025
top-4
```

## 3.4 输出

```text
outputs/agent_team_a_audit/
├── score_lineage_report.md
├── exp3_score_sources.csv
├── conformer_score_sources.csv
├── frozen_vs_rebuild_ensemble_comparison.csv
├── public_score_source_check.md
└── team_a_final_decision.md
```

## 3.5 结论格式

必须输出：

```text
EXP3_SCORE_SOURCE = frozen / rebuild / unknown
CURRENT_ENSEMBLE_SOURCE = ...
PUBLIC_ENSEMBLE_SOURCE = ...
RECOMMENDATION = use_current / switch_to_frozen_exp3 / cannot_determine
```

---

# 4. Agent Team B — Rank Ensemble Arena

## 4.1 目标

继续优化当前最佳：

```text
E3_rank w=0.50
BA=0.7917
```

重点：

1. 保持 fold2 的提升；
2. 修复 fold3 的下降；
3. 搜索更稳健的 rank ensemble 方法；
4. 生成新的 ensemble candidate。

## 4.2 输入

来自 Team A 确认后的 score：

```text
Conformer tuned original_trial scores
exp3 selected original_trial scores
```

## 4.3 方法

### B1. Weighted rank sweep

```text
rank_ens = w * rank_conformer + (1-w) * rank_exp3
w = 0.00 to 1.00, step=0.025
```

注意：

```text
rank 方向必须统一：分数越高越 positive，rank 越靠前。
```

### B2. Borda count

```text
rank_sum = rank_conformer + rank_exp3
rank_sum 最小的 4 个 → positive
```

### B3. Reciprocal rank

```text
score = w / (rank_conformer + eps) + (1-w) / (rank_exp3 + eps)
```

### B4. Percentile rank

每个 subject 内：

```text
rank_percentile = rank / 8
```

再加权融合。

### B5. Rank + margin

先计算每个模型 subject 内 margin：

```text
margin = abs(score - subject_median_score)
或 abs(score - 0.5)
```

然后：

```text
rank_score = rank_component + lambda * normalized_margin_component
lambda ∈ [0.05, 0.10, 0.20]
```

### B6. Stability objective

不仅按 BA 排序，也计算：

```text
stable_score = BA_mean - 0.25 * BA_std
min_fold_BA
folds_ge_current_best
fold3_BA
```

## 4.4 Promote 标准

新 rank ensemble 只有满足以下条件才替代当前 best：

```text
BA > 0.7917
或 BA >= 0.7917 且 fold3 改善且 std 更低
至少 3/5 folds >= current best
public 未参与选择
全局固定方法和权重
```

## 4.5 输出

```text
outputs/agent_team_b_rank_ensemble/
├── rank_ensemble_leaderboard.csv
├── per_fold_rank_ensemble.csv
├── rank_method_comparison.md
├── fold3_repair_analysis.md
├── public_submission_best_rank_ensemble.xlsx
├── public_submission_best_rank_ensemble_with_prob.xlsx
└── team_b_final_report.md
```

---

# 5. Agent Team C — Fold3 Ensemble Failure Diagnostic

## 5.1 目标

解释为什么当前 best ensemble 在 fold3 变差：

```text
Conformer fold3 = 0.7083
exp3 fold3 = 0.7500
ensemble fold3 = 0.6875
```

这说明 rank average 存在 fold3 负互补。

## 5.2 必须检查

1. fold3 test subjects 是谁？
2. 每个 subject 的 8 trial 排名情况。
3. 哪些 subject 被 ensemble 改坏？
4. 哪些 original_trial：
   - Conformer 排对；
   - exp3 排对；
   - ensemble 排错；
5. 是否是 top-4 边界 trial 被交换？
6. 错误是否集中在 DEP 或 HC？
7. 错误是否集中在某些 trial_id？
8. 是否由两个模型 rank 冲突导致？
9. 是否可以通过全局 rank 权重修复？
10. 是否存在适合 public 的可迁移规则？

## 5.3 禁止

禁止使用：

```text
per-fold 专用权重
per-subject 专用规则
per-trial hard rule
public feedback
```

诊断可以发现模式，但最终模型只能使用全局固定策略。

## 5.4 输出

```text
outputs/agent_team_c_fold3_diagnostic/
├── fold3_subject_error_table.csv
├── fold3_trial_rank_comparison.csv
├── fold3_rank_swap_cases.md
├── fold3_model_disagreement.md
├── fold3_repair_recommendation.md
└── team_c_final_report.md
```

## 5.5 结论格式

只能输出：

```text
fold3_issue = rank_conflict / score_scale / subject_specific / no_transferable_rule / unknown
recommendation = adjust_global_rank_weight / use_stable_objective / no_action
```

---

# 6. Agent Team D — Tuned Conformer Seed Ensemble

## 6.1 目标

训练 tuned Conformer 多 seed，降低单 seed 方差，再与 exp3 做 rank ensemble。

## 6.2 固定配置

```yaml
model: eeg_conformer_lite_tuned
dropout: 0.3
epochs: 70
emb_size: 32
depth: 2
num_heads: 4
patch_size: 32
lr: 0.0005
weight_decay: 0.01
patience: 14
batch_size: 16
```

## 6.3 Seeds

```text
42
2024
3407
1234
777
```

如果已有 seed 42 或已有 tuned Conformer 结果，可复用，但必须记录来源。

## 6.4 每个 seed 输出

```text
outputs/agent_team_d_conformer_seed/seed_{seed}/
├── fold_0/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
├── original_trial_scores.csv
├── top4_predictions_cv.csv
├── fold_metrics.csv
├── summary_metrics.json
├── public_original_trial_scores.csv
└── public_submission_seed_{seed}.xlsx
```

## 6.5 Ensemble 方法

### D1. Conformer seed probability average

```text
score = mean(score_seed_i)
```

### D2. Conformer seed rank average

```text
rank = mean(rank_seed_i)
```

### D3. Conformer median score

```text
score = median(score_seed_i)
```

### D4. Conformer seed ensemble + exp3 rank ensemble

```text
rank_combined = w * rank_conformer_seed_ensemble + (1-w) * rank_exp3
w = 0.00 to 1.00, step=0.025
```

## 6.6 Promote 标准

```text
BA > 0.7917
至少 3/5 folds >= current best
std 不明显增加
```

## 6.7 失败处理

如果某 seed OOM：

```text
batch_size 16 -> 8 -> 4
retry once
```

如果仍失败：

```text
mark seed_failed
continue next seed
```

至少 3 个完整 seeds 才生成主 ensemble。

## 6.8 输出

```text
outputs/agent_team_d_conformer_seed/
├── seed_results.csv
├── conformer_seed_ensemble_results.csv
├── conformer_seed_plus_exp3_results.csv
├── public_submission_best_seed_ensemble.xlsx
├── public_submission_best_seed_ensemble_with_prob.xlsx
└── team_d_final_report.md
```

---

# 7. Agent Team E — Fold2/Fold3 Targeted Conformer Tuning

## 7.1 目标

继续小范围调 tuned Conformer，重点修复：

```text
fold2 = 0.7292
fold3 = 0.7083
```

同时保持：

```text
fold4 = 0.7917
```

## 7.2 筛选 folds

只用：

```text
fold2
fold3
fold4
```

因为：

```text
fold2/fold3 是弱点
fold4 是强点，需要保持
```

## 7.3 小范围候选

不要全组合大网格。按顺序试：

### E1. Epoch sweep

```text
epochs = [65, 70, 75]
```

### E2. Dropout around best

```text
dropout = [0.25, 0.30, 0.35]
```

### E3. Weight decay

```text
weight_decay = [0.005, 0.01, 0.02]
```

### E4. Patch size

```text
patch_size = [16, 32, 64]
```

只有前面无提升时才试 patch size。

## 7.4 Screening 晋级 full 5-fold 条件

一个配置进入 full 5-fold 必须满足：

```text
fold2 >= 0.7292
fold3 >= 0.7500 或至少 fold3 提升 +0.0417
fold4 >= 0.7500
screening mean >= current tuned Conformer 在 fold2/3/4 的 mean
```

最多只允许 2 个配置进入完整 5-fold。

## 7.5 输出

```text
outputs/agent_team_e_conformer_tuning/
├── screening_results.csv
├── full5_candidates.csv
├── fold23_tuning_report.md
├── public_submission_best_fold23_conformer.xlsx
└── team_e_final_report.md
```

---

# 8. Agent Team F — Final Candidate Board

## 8.1 目标

汇总所有团队结果，选择最终 submission。

## 8.2 候选

必须比较：

```text
C0: exp3_vote_alpha062_cwmean_top4
C1: eeg_conformer_lite_tuned
C2: current Conformer+exp3 rank ensemble BA=0.7917
C3: Team B best rank ensemble
C4: Team D best seed ensemble
C5: Team E best tuned conformer
C6: Any Team D/E + exp3 ensemble
```

## 8.3 统一表

输出：

```text
outputs/agent_team_f_final_board/final_candidate_board.csv
```

字段：

```csv
candidate,source,ba,mf1,ba_std,fold0,fold1,fold2,fold3,fold4,folds_ge_current_best,submission_path,sha256,decision,reason
```

## 8.4 最终选择规则

最终主提交：

```text
BA 最高
且 folds_ge_current_best >= 3
且 std 不明显增加
且 submission 审计通过
且 public 未参与选择
```

如果没有新候选超过 0.7917：

```text
保持 current Conformer+exp3 rank ensemble
```

## 8.5 提交审计

最终 selected submission 必须检查：

```text
80 rows
10 users
8 trials/user
4 positive/user
4 neutral/user
no duplicates
no missing
Emotion_label ∈ {0,1}
columns match template
xlsx readable
```

## 8.6 输出

```text
outputs/agent_team_f_final_board/
├── final_candidate_board.csv
├── final_submission_selected.xlsx
├── final_submission_selected_with_prob.xlsx
├── final_submission_audit.md
├── final_recommendation.md
├── team_update.md
└── final_hashes.txt
```

---

# 9. 任务执行顺序

推荐并行顺序：

```text
Step 1: Team A 立即启动，先查 score lineage
Step 2: Team B 等 Team A 的 score 来源确认后跑 rank ensemble arena
Step 3: Team C 同步分析 fold3
Step 4: Team D 后台开始 Conformer 多 seed
Step 5: Team E 在 GPU 空闲或 Team D 未占满时做 fold2/fold3 小调
Step 6: Team F 最后汇总
```

如果只能串行：

```text
A → B → C → D → E → F
```

如果时间很多：

```text
A + C 可并行
D 后台长跑
B 做快速后处理
E 用剩余 GPU 做筛选
F 最后收尾
```

---

# 10. 最终终端摘要格式

所有团队结束后打印：

```text
===== Agent Teams Final Summary =====

Current best before teams:
  Conformer + exp3 rank ensemble
  BA:
  MF1:

Team A lineage:
  exp3 source:
  current ensemble source:
  decision:

Team B rank ensemble:
  best method:
  BA:
  MF1:
  fold3:
  decision:

Team C fold3 diagnostic:
  issue:
  repair:
  decision:

Team D conformer seed:
  seeds completed:
  best ensemble:
  BA:
  decision:

Team E conformer tuning:
  best config:
  BA:
  decision:

Final selected:
  model:
  BA:
  MF1:
  submission:
  SHA256:

Recommendation:
  SUBMIT_CURRENT_ENSEMBLE / SUBMIT_NEW_ENSEMBLE / SUBMIT_CONFORMER / SUBMIT_EXP3_BACKUP

Do not continue:
  External data / AdaBN / DANN / GraphTransformer / old contrastive

=================================
```

---

# 11. 最后执行原则

这次 Agent Teams 的目标不是证明每条路线都强，而是围绕当前最强 ensemble 做：

```text
1. 分数来源确认；
2. rank ensemble 继续优化；
3. fold3 负互补修复；
4. tuned Conformer 多 seed 降方差；
5. fold2/fold3 小调；
6. 最终候选决策。
```

如果所有新尝试都不超过：

```text
BA = 0.7917
```

则最终保持：

```text
Conformer + exp3 rank ensemble E3_rank w=0.50
```

作为主提交。
