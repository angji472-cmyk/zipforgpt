# 7小时无人值守计划：Final EEG 模型可信重建 + 受控调优探索 + Seed Ensemble

## 0. 核心目标

当前最强模型为：

```text
exp3_vote_alpha062_cwmean_top4
```

当前已知配置：

```text
Architecture: DualBranchEEGNet Wide
Time branch: raw EEG branch
Frequency branch: FFT / log_power branch
Fusion: prob = alpha * prob_time + (1 - alpha) * prob_freq
Alpha: 0.62
Aggregation: confidence_weighted_mean
Post-processing: subject-level top-4
Prediction level: original_trial
Current CV BA: 0.7583 ± 0.0618
Current CV MF1: 0.7583
Frozen submission: outputs/final_review/final_submission_clean.xlsx
Frozen submission SHA256: bafdb4212a0685eadfbde10e9d36019dee8ae4b1d7e9d55a8b0c2c1f53c28b92
```

本计划目标：

1. 保留当前 frozen final submission 作为安全底线。
2. 重建可复现 final pipeline。
3. 在不跑偏的前提下加入较多模型调优探索。
4. 优先围绕已经证明有效的 `exp3_vote` 做调优，而不是重新开失败路线。
5. 尝试同结构 seed ensemble。
6. 尝试有限、受控、可解释的超参数微调。
7. 输出最终候选排名、提交文件、审计报告。
8. 保证即使所有新实验失败，也不会破坏当前最好结果。

---

## 1. 最高优先级禁令

以下内容禁止执行：

1. 禁止覆盖当前 frozen submission：

```text
outputs/final_review/final_submission_clean.xlsx
```

2. 禁止覆盖已有最终报告：

```text
outputs/final_review/final_trustworthiness_report.md
outputs/final_model_summary.md
outputs/final_review/repro_lock/final_reproducibility_lock_report.md
```

3. 禁止使用 public test label。
4. 禁止用 public 反馈调参。
5. 禁止重新打开以下旧路线作为主线：
   - AdaBN
   - DANN
   - ST-DADGAT-lite
   - GraphTransformer
   - Robot Faces external pretraining
   - DENS external pretraining
   - Voice-User external pretraining
   - FEEL
   - MMD
   - 大规模 SupCon
   - 外部数据混合训练
6. 禁止 per-fold oracle alpha 作为最终方案。
7. 禁止 per-subject alpha。
8. 禁止 per-fold threshold。
9. 禁止 stacking / meta-classifier。
10. 禁止修改 public submission 的 user_id / trial_id 顺序逻辑。
11. 禁止删除旧实验目录。
12. 禁止移动旧实验目录。
13. 禁止把单 fold 提升包装成主线提升。
14. 禁止无审计地替换最终提交文件。

---

## 2. 允许探索的范围

本次 7 小时只允许围绕以下方向探索：

### 2.1 可复现性

```text
checkpoint → branch_probs → alpha=0.62 → aggregation → top-4 → submission
```

### 2.2 后处理微调

仅允许：

```text
alpha sweep around 0.62
aggregation sweep
top-4 invariant check
mean_prob / cw_mean backup
rank aggregation
```

### 2.3 同结构 seed ensemble

仅允许同一个结构：

```text
DualBranchEEGNet Wide
raw branch + FFT branch
```

不同 seed，概率平均或 rank average。

### 2.4 小规模结构/训练超参数探索

只允许轻量微调，不允许换路线：

```text
dropout small sweep
learning rate small sweep
weight decay small sweep
FFT preprocessing variants
branch fusion temperature / calibration
branch contribution diagnostics
```

### 2.5 模型校准

允许：

```text
global temperature scaling
branch temperature scaling
probability sharpening / smoothing
```

但必须在 CV 上选择，且明确标记 model-selection bias。

---

## 3. 总体时间预算

总预算约 7 小时。请按以下顺序执行。

| Phase | 内容 | 预算 |
|---|---:|---:|
| 0 | 安全快照与目录准备 | 10 min |
| 1 | Deterministic rebuild | 60–90 min |
| 2 | 后处理复核与小 sweep | 30–45 min |
| 3 | 同结构 seed ensemble | 3–4 h |
| 4 | 轻量模型调优探索 | 1–1.5 h |
| 5 | 候选统一评估与选择 | 30 min |
| 6 | 最终审计与报告 | 30 min |
| 7 | 剩余时间容错补跑 | remaining |

如果时间不够，按优先级保留：

```text
deterministic rebuild > seed ensemble > post-processing sweep > model hyperparam exploration
```

---

## 4. 输出根目录

所有新结果写到：

```text
outputs/final_7h_tuning_run/
```

建议结构：

```text
outputs/final_7h_tuning_run/
├── 00_safety_snapshot/
├── 01_deterministic_rebuild/
├── 02_postprocess_sweeps/
├── 03_seed_ensemble/
├── 04_light_model_tuning/
├── 05_candidate_selection/
├── 06_final_audit/
├── 07_final_reports/
└── run_state/
```

每个 phase 完成后更新：

```text
outputs/final_7h_tuning_run/run_state/progress.md
```

---

## 5. Phase 0：安全快照

### 5.1 创建快照索引

生成：

```text
outputs/final_7h_tuning_run/00_safety_snapshot/frozen_artifacts_index.md
outputs/final_7h_tuning_run/00_safety_snapshot/frozen_artifacts_hashes.txt
```

至少 hash：

```text
outputs/final_review/final_submission_clean.xlsx
outputs/final_review/final_submission_with_prob_backup.xlsx
outputs/final_review/repro_lock/final_reproducibility_lock_report.md
outputs/final_review/repro_lock/final_artifact_hashes.txt
outputs/final_model_summary.md
outputs/exp3_aggregation_top4_sweep/per_fold_aggregation_top4.csv
outputs/exp3_alpha_top4_sweep/alpha_top4_report.md
outputs/exp3_aggregation_top4_sweep/aggregation_top4_report.md
```

### 5.2 写保护声明

生成：

```text
outputs/final_7h_tuning_run/00_safety_snapshot/do_not_overwrite_notice.md
```

写明：

```text
Current frozen final is the safety baseline.
No generated file may overwrite it.
All new results are candidates only.
```

---

## 6. Phase 1：Deterministic Rebuild

## 6.1 目的

把当前 `LOCKED_WITH_CAVEAT` 尽可能升级为：

```text
LOCKED
```

也就是：

```text
checkpoint → inference → branch_probs → alpha/aggregation/top4 → CV/submission
```

能连续两次跑出相同结果。

## 6.2 创建唯一入口脚本

新建：

```text
scripts/rebuild_final_exp3_alpha062_cwmean_top4.py
```

运行两次：

```bash
python scripts/rebuild_final_exp3_alpha062_cwmean_top4.py \
  --run_id run_a \
  --output_dir outputs/final_7h_tuning_run/01_deterministic_rebuild/run_a \
  --device cuda \
  --deterministic true
```

```bash
python scripts/rebuild_final_exp3_alpha062_cwmean_top4.py \
  --run_id run_b \
  --output_dir outputs/final_7h_tuning_run/01_deterministic_rebuild/run_b \
  --device cuda \
  --deterministic true
```

## 6.3 确定性设置

必须设置：

```python
import os
import random
import numpy as np
import torch

SEED = 20260427
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
```

DataLoader：

```python
shuffle = False
num_workers = 0 or 1
persistent_workers = False
pin_memory = True
```

Inference：

```python
model.eval()
with torch.no_grad():
    ...
```

必须记录：

```text
torch version
cuda version
cudnn version
gpu name
python version
git commit if available
```

## 6.4 输出文件

每次 run 输出：

```text
branch_probs_window.csv
trial10s_scores.csv
original_trial_scores.csv
top4_predictions_cv.csv
fold_metrics.csv
summary_metrics.json
public_original_trial_scores.csv
public_submission_rebuilt.xlsx
public_submission_rebuilt_with_prob.xlsx
environment.txt
run_metadata.json
```

### branch_probs_window.csv 字段

```csv
fold,subject_id,original_trial_id,trial10s_id,window_id,y_true,prob_time,prob_freq,prob_alpha062
```

### original_trial_scores.csv 字段

```csv
fold,subject_id,original_trial_id,y_true,score_mean_prob,score_cw_mean,final_score
```

### top4_predictions_cv.csv 字段

```csv
fold,subject_id,original_trial_id,y_true,final_score,pred
```

### public_original_trial_scores.csv 字段

```csv
user_id,trial_id,score_mean_prob,score_cw_mean,final_score
```

---

## 7. Phase 1.5：Rebuild 比较

创建脚本：

```text
scripts/compare_deterministic_rebuild_runs.py
```

比较 run_a 与 run_b：

```bash
python scripts/compare_deterministic_rebuild_runs.py \
  --run_a outputs/final_7h_tuning_run/01_deterministic_rebuild/run_a \
  --run_b outputs/final_7h_tuning_run/01_deterministic_rebuild/run_b \
  --frozen_submission outputs/final_review/final_submission_clean.xlsx \
  --output_dir outputs/final_7h_tuning_run/01_deterministic_rebuild/comparison
```

必须比较：

1. `branch_probs_window.csv`
2. `original_trial_scores.csv`
3. `top4_predictions_cv.csv`
4. public submission
5. CV metrics
6. frozen submission 差异

输出：

```text
outputs/final_7h_tuning_run/01_deterministic_rebuild/comparison/determinism_comparison.md
outputs/final_7h_tuning_run/01_deterministic_rebuild/comparison/branch_prob_diff.csv
outputs/final_7h_tuning_run/01_deterministic_rebuild/comparison/public_submission_diff.csv
outputs/final_7h_tuning_run/01_deterministic_rebuild/reports/deterministic_rebuild_report.md
```

判定：

```text
LOCKED:
run_a == run_b labels exactly, deterministic generation successful

NEW_LOCKED_CANDIDATE:
run_a == run_b but differs from old frozen submission

NOT_DETERMINISTIC:
run_a != run_b labels or top4 rankings unstable
```

---

## 8. Phase 2：后处理复核与小 sweep

此阶段不训练模型，只使用已有或重建出的 branch probabilities。

## 8.1 Alpha 局部复核

当前 best：

```text
alpha = 0.62
```

只允许局部 sweep：

```text
alpha ∈ [0.58, 0.66], step=0.005
```

固定：

```text
aggregation = confidence_weighted_mean
top-4 = enabled
```

输出：

```text
outputs/final_7h_tuning_run/02_postprocess_sweeps/alpha_local_sweep.csv
outputs/final_7h_tuning_run/02_postprocess_sweeps/alpha_local_sweep_report.md
```

Promote 规则：

```text
BA > 0.7583
at least 3/5 folds >= current final
std not higher by > 0.02
```

如果没有达到，不替换 alpha=0.62。

## 8.2 Aggregation 复核

固定：

```text
alpha = 0.62
top-4 = enabled
```

只允许以下 aggregation：

```text
mean_prob
confidence_weighted_mean
mean_logit
median_prob
trimmed_mean_20
rank_average
```

禁止再试复杂可学习 aggregation。

输出：

```text
outputs/final_7h_tuning_run/02_postprocess_sweeps/aggregation_recheck.csv
outputs/final_7h_tuning_run/02_postprocess_sweeps/aggregation_recheck_report.md
```

## 8.3 Backup submission

必须生成两个 backup：

```text
A: alpha=0.62 + confidence_weighted_mean + top4
B: alpha=0.62 + mean_prob + top4
```

输出：

```text
outputs/final_7h_tuning_run/02_postprocess_sweeps/submission_cwmean_top4.xlsx
outputs/final_7h_tuning_run/02_postprocess_sweeps/submission_meanprob_top4.xlsx
```

---

## 9. Phase 3：同结构 Seed Ensemble

这是本计划中最主要的提升尝试。

## 9.1 训练 seeds

按优先级：

```text
seed_42
seed_2024
seed_3407
seed_1234
seed_777
```

如果已有 seed_42 对应原始 `exp3_vote_5fold`，可以复用，但必须记录来源。

## 9.2 固定模型结构

禁止改结构，只允许 seed 改变。

必须固定：

```text
DualBranchEEGNet Wide
F1=16
F2=32
dropout 与原 exp3 一致
time branch raw EEG
freq branch FFT/log_power
same split
same preprocessing
same optimizer unless原配置不可读取
```

如果原配置可读，严格复用原配置。  
如果不可读，写明 fallback，并尽量与原 exp3 保持一致。

## 9.3 训练输出

每个 seed：

```text
outputs/final_7h_tuning_run/03_seed_ensemble/seed_{seed}/
```

每个 seed 输出：

```text
fold_0/best.pt
fold_1/best.pt
fold_2/best.pt
fold_3/best.pt
fold_4/best.pt
branch_probs_window.csv
original_trial_scores.csv
top4_predictions_cv.csv
public_original_trial_scores.csv
public_submission_seed_{seed}.xlsx
fold_metrics.csv
summary_metrics.json
train_logs/
```

## 9.4 8GB 显存安全配置

默认：

```yaml
batch_size: 16
mixed_precision: true
num_workers: 2
pin_memory: true
persistent_workers: false
gradient_clip_norm: 1.0
```

OOM 后自动：

```text
batch_size 16 -> 8
clear cuda cache
retry current fold once
```

每 fold 结束：

```python
del model, optimizer, dataloader
torch.cuda.empty_cache()
gc.collect()
```

## 9.5 训练失败策略

如果某 seed 某 fold 失败：

1. 自动重试一次。
2. 仍失败则标记失败。
3. 不停止整个 7 小时计划。
4. 至少 3 个完整 seeds 才生成 ensemble。
5. 少于 3 个完整 seeds，只报告 partial，不替代 final。

---

## 10. Phase 4：Seed Ensemble 评估

创建：

```text
scripts/evaluate_exp3_seed_ensemble.py
```

## 10.1 单 seed 评估

每个 seed 固定：

```text
alpha=0.62
aggregation=confidence_weighted_mean
top-4
```

输出：

```text
seed_results.csv
```

字段：

```csv
seed,ba_mean,ba_std,mf1_mean,mf1_std,fold0_ba,fold1_ba,fold2_ba,fold3_ba,fold4_ba,decision
```

## 10.2 Ensemble 方法

只允许三种：

### E1：probability average

```text
score = mean(seed_scores)
```

### E2：rank average

```text
每个 subject 内每个 seed 排 rank
平均 rank
top-4
```

### E3：median probability

```text
score = median(seed_scores)
```

禁止：

```text
per-fold weights
per-subject weights
stacking
validation-trained meta-model
```

输出：

```text
outputs/final_7h_tuning_run/03_seed_ensemble/ensemble/ensemble_results.csv
outputs/final_7h_tuning_run/03_seed_ensemble/ensemble/per_fold_ensemble_comparison.csv
outputs/final_7h_tuning_run/03_seed_ensemble/ensemble/public_submission_ensemble.xlsx
outputs/final_7h_tuning_run/03_seed_ensemble/reports/seed_ensemble_report.md
```

Promote 规则：

```text
BA > 0.7583
MF1 >= 0.7583
at least 3/5 folds >= final
std not significantly worse
generation deterministic
```

否则不替换 final。

---

## 11. Phase 5：轻量模型调优探索

只有在 seed ensemble 至少启动后，且还有时间时执行。  
此阶段必须是小规模、受控、围绕 exp3 的调优。

输出目录：

```text
outputs/final_7h_tuning_run/04_light_model_tuning/
```

## 11.1 调优优先级

按顺序尝试，前一个没有希望再继续下一个。

### T1：dropout 小 sweep

只试：

```text
dropout ∈ [0.30, 0.35, 0.40]
```

如果原 exp3 已经是 0.35，则只新增：

```text
0.30
0.40
```

训练策略：

```text
只先跑 fold_1 和 fold_4
```

原因：

```text
fold_1 容易，fold_4 难
同时看是否过拟合和是否救困难 fold
```

继续条件：

```text
fold_1 不低于 final 对应 fold
且 fold_4 提升 >= 0.02
```

不满足则停止 dropout sweep。

### T2：learning rate 小 sweep

只试：

```text
lr ∈ [0.0003, 0.0005, 0.0008]
```

同样先跑 fold_1 和 fold_4。

继续条件同上。

### T3：weight decay 小 sweep

只试：

```text
weight_decay ∈ [0.005, 0.01, 0.02]
```

先跑 fold_1 和 fold_4。

### T4：FFT 分支温和变体

只允许非常轻量的频域变体，不重写数据管线：

```text
current: log_power
variant_1: log1p_power
variant_2: zscore_log_power train_fold_only
```

禁止引入新 STFT / Graph / Transformer。

先跑 fold_1 和 fold_4。

### T5：branch temperature scaling

不训练模型，只调分支概率温度：

```text
prob_time = softmax(logits_time / T_time)
prob_freq = softmax(logits_freq / T_freq)

T_time, T_freq ∈ {0.8, 1.0, 1.2}
alpha fixed 0.62
top-4 enabled
```

如果 logits 不存在，只跳过，不重新 inference 大量内容。

## 11.2 轻量调优的晋级规则

任何 T 类实验要进入 full 5-fold，必须满足：

```text
fold_1 >= final fold_1 - 0.01
fold_4 >= final fold_4 + 0.02
无 OOM
无 pipeline 改动
```

进入 full 5-fold 后，promote 必须满足：

```text
BA > 0.7583
at least 3/5 folds >= final
std not worse
```

否则只记录，不采用。

## 11.3 输出

```text
outputs/final_7h_tuning_run/04_light_model_tuning/tuning_screening.csv
outputs/final_7h_tuning_run/04_light_model_tuning/tuning_full_results.csv
outputs/final_7h_tuning_run/04_light_model_tuning/light_tuning_report.md
```

---

## 12. Phase 6：候选统一评估

统一比较以下候选：

```text
Frozen final
Deterministic rebuild
Postprocess local best
Seed ensemble probability average
Seed ensemble rank average
Seed ensemble median
Any promoted light tuning candidate
Mean_prob backup
```

生成：

```text
outputs/final_7h_tuning_run/05_candidate_selection/candidate_summary.csv
outputs/final_7h_tuning_run/05_candidate_selection/candidate_decision.md
```

字段：

```csv
candidate_name,source,cv_ba,cv_mf1,cv_std,folds_ge_frozen,submission_path,sha256,decision,reason
```

决策：

```text
SUBMIT_FROZEN
SUBMIT_REBUILD
SUBMIT_ENSEMBLE
SUBMIT_TUNED
```

选择规则：

1. 如果没有候选严格超过 frozen final，选择 frozen final。
2. 如果候选超过但 folds_ge_frozen < 3，不替换。
3. 如果候选超过但 submission 不 deterministic，不替换。
4. 如果 ensemble 提升且稳定，选择 ensemble。
5. 如果两个候选接近，选 std 更低、工程风险更小的。

---

## 13. Phase 7：最终提交审计

最终选择后生成：

```text
outputs/final_7h_tuning_run/06_final_audit/final_submission_selected.xlsx
outputs/final_7h_tuning_run/06_final_audit/final_submission_selected_with_prob_backup.xlsx
outputs/final_7h_tuning_run/06_final_audit/final_submission_audit.md
outputs/final_7h_tuning_run/06_final_audit/final_submission_check.csv
```

必须检查：

1. 80 行。
2. 10 个 user。
3. 每 user 8 trials。
4. 每 user 4 个 1、4 个 0。
5. trial_id 无重复。
6. trial_id 类型 int。
7. user_id 类型 string。
8. Emotion_label 只含 0/1。
9. Excel 可读。
10. 没有多余列。
11. 若模板不允许 prob，clean 文件必须不含 prob。

---

## 14. Phase 8：最终报告

生成：

```text
outputs/final_7h_tuning_run/07_final_reports/final_7h_tuning_run_report.md
outputs/final_7h_tuning_run/07_final_reports/final_recommendation.md
outputs/final_7h_tuning_run/07_final_reports/team_update.md
```

### 14.1 final_7h_tuning_run_report.md

必须包含：

1. 任务目标；
2. 当前 frozen final；
3. deterministic rebuild 结果；
4. postprocess 复核；
5. seed ensemble 结果；
6. light tuning 结果；
7. candidate selection；
8. final submission；
9. 风险；
10. 后续建议。

### 14.2 final_recommendation.md

必须回答：

```text
最终推荐提交哪个文件？
为什么？
备份文件是什么？
是否还有 caveat？
```

### 14.3 team_update.md

写给队友，语言清晰：

```markdown
# 队内更新

## 当前最终推荐
...

## 今天 7 小时跑了什么
...

## 哪些路线仍然不建议继续
...

## 是否有新候选超过 frozen final
...

## 最终要提交哪个文件
...
```

---

## 15. 进度日志

每个 phase 完成后追加：

```text
outputs/final_7h_tuning_run/run_state/progress.md
```

格式：

```markdown
## Phase X completed at YYYY-MM-DD HH:MM

Status:
- completed / partial / failed

Key outputs:
- ...

Metrics:
- ...

Issues:
- ...

Next:
- ...
```

如果任务中断，重启后必须先读：

```text
outputs/final_7h_tuning_run/run_state/progress.md
```

然后从未完成 phase 继续。

---

## 16. 失败与降级策略

### 16.1 Deterministic rebuild 失败

如果 `NOT_DETERMINISTIC`：

```text
不替换 frozen final
继续 seed ensemble
记录 caveat
```

### 16.2 Seed ensemble 没跑完

如果完成 seeds < 3：

```text
不生成主 ensemble
只报告 partial
不替代 frozen final
```

如果完成 seeds >= 3：

```text
生成 ensemble candidate
但必须过 promote 规则
```

### 16.3 Light tuning 不稳定

任何调优实验只提升 1 个 fold：

```text
mark supplement
do not promote
```

### 16.4 OOM

自动降 batch：

```text
16 → 8
```

仍 OOM：

```text
skip current candidate
continue next phase
```

### 16.5 时间耗尽

如果时间接近 7 小时：

1. 停止启动新训练。
2. 完成当前 fold。
3. 生成 partial report。
4. 保留 frozen final 推荐。

---

## 17. 最终终端摘要

任务结束时打印：

```text
===== 7h Final EEG Tuning Run Summary =====

Frozen final:
  path:
  SHA256:
  BA:
  MF1:

Deterministic rebuild:
  verdict:
  BA:
  MF1:
  same as frozen:
  notes:

Postprocess recheck:
  best alpha:
  best aggregation:
  improvement:

Seed ensemble:
  seeds completed:
  best ensemble:
  BA:
  MF1:
  folds >= frozen:
  decision:

Light tuning:
  candidates screened:
  candidates promoted:
  best tuned:
  decision:

Selected final submission:
  path:
  SHA256:
  rows:
  subjects:
  4/4 constraint:
  audit:

Recommendation:
  SUBMIT_FROZEN / SUBMIT_REBUILD / SUBMIT_ENSEMBLE / SUBMIT_TUNED

Do not continue:
  AdaBN / DANN / GraphTransformer / external pretraining / contrastive large grid

Next low-cost step if more time remains:
  additional same-structure seed only

====================================
```

---

## 18. 最后执行原则

本计划允许“很多调优探索”，但只允许在以下边界内：

```text
same model family
same split
same final postprocess
no public tuning
no old failed routes
no large architecture change
no external data
```

本次 7 小时真正要做的是：

```text
1. 把当前最强方案做成可复现工程产物；
2. 用同结构 seed ensemble 和小规模超参微调榨最后一点分；
3. 保证任何新结果失败时都能安全回退到 frozen final。
```
