# 长时间无人值守计划：EEG 新模型竞技场 + 最终模型保底 + 严格闸门

## 0. 任务背景

当前最强已知方案：

```text
Final safety baseline:
exp3_vote_alpha062_cwmean_top4

Architecture:
- DualBranchEEGNet Wide
- raw/time branch
- FFT/log_power frequency branch
- fixed global alpha = 0.62
- aggregation = confidence_weighted_mean
- postprocess = subject-level top-4
- prediction level = original_trial

Known CV:
BA = 0.7583 ± 0.0618
MF1 = 0.7583

Frozen submission:
outputs/final_review/final_submission_clean.xlsx

Frozen SHA256:
bafdb4212a0685eadfbde10e9d36019dee8ae4b1d7e9d55a8b0c2c1f53c28b92
```

本计划允许开启多个新模型做受控探索，但必须严格防止跑偏：所有新模型都必须在统一 split、统一 original_trial 粒度、统一 subject top-4 口径下和当前 final 做比较。

---

## 1. 总目标

本次任务目标不是只生成报告，而是实际跑一个长时间新模型竞技场：

1. 保留 `exp3_vote_alpha062_cwmean_top4` 作为永远不覆盖的安全底线。
2. 先完成 deterministic rebuild 和已启动 seed ensemble。
3. 开启多个新模型候选：
   - TSception-lite
   - EEG-Conformer-lite
   - LGGNet-lite
   - EEG-Deformer-lite
   - EmT-lite
   - DAMGCN-lite / channel-band graph attention
   - exp3 same-structure seed ensemble
4. 每个模型先做快速筛选 fold。
5. 只有通过筛选的模型才跑完整 5-fold。
6. 所有模型输出统一 original_trial + top-4 指标。
7. 最终只允许 promote 满足严格条件的模型。
8. 如果没有任何新模型超过 final，就保持 frozen final。
9. 生成完整可给队友看的模型竞技场报告。

---

## 2. 最高优先级约束

### 2.1 禁止事项

以下行为绝对禁止：

1. 禁止覆盖：

```text
outputs/final_review/final_submission_clean.xlsx
```

2. 禁止覆盖旧最终报告。
3. 禁止使用 public test label。
4. 禁止用 public 反馈调参。
5. 禁止把 public 预测结果用于模型选择。
6. 禁止 per-fold oracle alpha。
7. 禁止 per-subject alpha。
8. 禁止 per-fold threshold。
9. 禁止 stacking / meta-classifier。
10. 禁止外部数据混合监督训练。
11. 禁止重开以下已判定无主线价值的路线：
    - AdaBN
    - DANN
    - Robot Faces external pretraining
    - DENS external pretraining
    - Voice-User external pretraining
    - FEEL external pretraining
    - 大规模 SupCon 网格
    - MMD
12. 禁止只因单个 fold 高就 promote。
13. 禁止只跑一个模型失败后提前结束。
14. 禁止“生成报告”代替“执行实验”。
15. 禁止删除旧结果。
16. 禁止移动旧结果。
17. 禁止覆盖旧输出目录。
18. 禁止把未完成 5-fold 的模型写成最终模型。

### 2.2 允许事项

允许：

1. 新建目录。
2. 训练新模型。
3. 小规模超参搜索。
4. 同结构 seed ensemble。
5. 新模型和 final 做低权重 ensemble 诊断。
6. 失败后自动跳过继续下一个模型。
7. 8GB 显存安全降级。
8. 局部 fold 筛选。
9. 完整 5-fold 复核。
10. 生成多个候选提交文件，但 final selected submission 必须经过审计。

---

## 3. 统一输出目录

所有新结果写入：

```text
outputs/long_model_arena/
```

目录结构：

```text
outputs/long_model_arena/
├── 00_safety/
├── 01_rebuild_and_seed_ensemble/
├── 02_model_adapters/
├── 03_screening/
│   ├── tsception_lite/
│   ├── eeg_conformer_lite/
│   ├── lggnet_lite/
│   ├── eeg_deformer_lite/
│   ├── emt_lite/
│   └── damgcn_lite/
├── 04_full_5fold/
├── 05_ensembles/
├── 06_candidate_selection/
├── 07_submissions/
├── 08_reports/
└── run_state/
```

每个阶段都要更新：

```text
outputs/long_model_arena/run_state/progress.md
```

---

## 4. 长时间运行纪律

### 4.1 不允许提前结束

除非出现以下情况之一，否则不要提前结束：

1. 所有候选模型都完成筛选或被明确跳过。
2. 至少 3 个新模型完成完整筛选。
3. 至少 1 个通过筛选的模型完成完整 5-fold。
4. 当前运行时间达到用户设定上限。
5. GPU 长时间不可用且无法恢复。
6. 代码库损坏风险。

禁止出现：

```text
只跑 deterministic rebuild
只跑一个 seed
只生成报告
然后宣布完成
```

### 4.2 GPU 空闲策略

如果 GPU 空闲超过 5 分钟：

1. 检查当前 phase 是否还有未跑任务。
2. 启动下一个候选模型。
3. 如果训练任务在后台，前台准备下一阶段脚本。
4. 如果一个模型 OOM，降级后重试一次。
5. 如果仍失败，记录失败并继续下一个模型。

### 4.3 时间利用策略

如果时间非常充裕：

1. 完成 seed ensemble 至少 3 seeds。
2. 跑完 TSception-lite 5-fold 或至少筛选。
3. 跑完 EEG-Conformer-lite 筛选。
4. 跑完 LGGNet-lite 筛选。
5. 跑一个最有希望的新模型完整 5-fold。
6. 再做 ensemble 诊断。

---

## 5. 统一评估口径

所有模型都必须转换到统一口径：

```text
original_trial level
subject-level top-4
每 subject 8 trial，top 4 score → class 1，其余 → class 0
```

所有模型必须输出：

```text
window_scores.csv 或 trial_scores.csv
original_trial_scores.csv
top4_predictions_cv.csv
fold_metrics.csv
summary_metrics.json
public_original_trial_scores.csv
public_submission.xlsx
```

每个模型使用统一后处理：

```text
default aggregation = confidence_weighted_mean if window-level probabilities available
fallback aggregation = mean_prob
postprocess = subject top-4
```

如果模型直接输出 original_trial probability，则直接 top-4。

---

## 6. Reference final fold metrics

当前 final reference：

```text
final_model = exp3_vote_alpha062_cwmean_top4
BA_mean = 0.7583
MF1_mean = 0.7583
```

已知 final per-fold BA 参考：

```text
fold_0 ≈ 0.7292
fold_1 ≈ 0.8542
fold_2 ≈ 0.7708
fold_3 ≈ 0.7500
fold_4 ≈ 0.6875
```

如果本地权威 CSV 中数值不同，以：

```text
outputs/exp3_aggregation_top4_sweep/per_fold_aggregation_top4.csv
```

为准。

---

## 7. 模型候选来源与依据

本计划参考以下公开方向：

1. TSception：面向 EEG emotion recognition 的多尺度时间卷积与空间不对称建模。
2. LGGNet：local-global-graph representations，用于 BCI / EEG 分类。
3. EEG-Conformer：compact convolutional transformer，结合局部卷积和全局自注意力。
4. EmT：cross-subject EEG emotion recognition 的 graph + transformer 方向。
5. EEG-Deformer / dense convolutional transformer：卷积和 Transformer 结合的 EEG decoding 方向。
6. DAMGCN / graph convolution + dual attention：通道图和频带注意力方向。
7. EEGPT / foundation model：只做可行性检查，不默认纳入主训练，因为迁移成本高。

注意：
这些方向只作为候选，不代表一定适合当前数据。  
最终只看本项目 CV + top-4 结果。

---

## 8. Phase 0：安全快照

生成：

```text
outputs/long_model_arena/00_safety/frozen_artifacts_hashes.txt
outputs/long_model_arena/00_safety/frozen_artifacts_index.md
outputs/long_model_arena/00_safety/do_not_overwrite_notice.md
```

Hash 以下文件：

```text
outputs/final_review/final_submission_clean.xlsx
outputs/final_review/final_submission_with_prob_backup.xlsx
outputs/final_review/repro_lock/final_reproducibility_lock_report.md
outputs/final_model_summary.md
outputs/exp3_aggregation_top4_sweep/per_fold_aggregation_top4.csv
outputs/exp3_alpha_top4_sweep/alpha_top4_report.md
outputs/exp3_aggregation_top4_sweep/aggregation_top4_report.md
```

---

## 9. Phase 1：完成 deterministic rebuild 与 seed ensemble 基础

这一步延续之前未完成任务。

### 9.1 deterministic rebuild

如果已有：

```text
outputs/final_7h_tuning_run/01_deterministic_rebuild/
```

读取并总结，不必重复跑。

如果没有完整结果，则重新跑：

```text
scripts/rebuild_final_exp3_alpha062_cwmean_top4.py
```

要求：

```text
run_a == run_b
记录是否和 frozen submission 一致
```

### 9.2 seed ensemble 补完

继续之前未完成的 seed：

```text
seed_3407
seed_1234
seed_777
```

至少要凑齐 3 个完整 seeds：

```text
seed_42
seed_2024
seed_3407
```

如果 seed_42 是原始 exp3 checkpoint，则可复用，但必须记录来源。

输出：

```text
outputs/long_model_arena/01_rebuild_and_seed_ensemble/seed_status.csv
outputs/long_model_arena/01_rebuild_and_seed_ensemble/seed_ensemble_report.md
```

### 9.3 seed ensemble 方法

只允许：

```text
probability average
rank average
median probability
```

禁止 stacking。

Promote 条件：

```text
BA > 0.7583
至少 3/5 folds >= final
std 不明显增加
```

---

## 10. Phase 2：构建模型适配器

在开始训练新模型前，先写统一 model adapter 接口。

创建：

```text
models/arena/
├── base_arena_model.py
├── tsception_lite.py
├── eeg_conformer_lite.py
├── lggnet_lite.py
├── eeg_deformer_lite.py
├── emt_lite.py
├── damgcn_lite.py
└── model_registry.py
```

每个模型必须实现：

```python
forward(x) -> logits
extract_features(x) -> embedding, optional
predict_proba(x) -> prob
```

统一输入规范：

### Raw EEG models

```text
x_raw: [B, C=30, T]
```

用于：

```text
TSception-lite
EEG-Conformer-lite
EEG-Deformer-lite
MS-1D-CNN variants
```

### Feature graph models

```text
x_feat: [B, C=30, T_bins=8, F]
```

用于：

```text
LGGNet-lite
DAMGCN-lite
EmT-lite
```

### Dual input models

可选：

```text
x_raw + x_fft
```

用于：

```text
exp3 variants only
```

禁止为了新模型重写整个数据集切分逻辑。

---

## 11. Phase 3：模型筛选设计

### 11.1 筛选 folds

为了节约时间，每个新模型先跑两个 folds：

```text
fold_1: easy/high-performing fold
fold_4: hard/low-performing fold
```

如果时间允许，加 fold_3：

```text
fold_3: another hard fold
```

筛选的理由：

```text
fold_1 检查上限能力
fold_4 检查困难 subject 泛化
fold_3 检查是否普遍稳健
```

### 11.2 筛选通过规则

一个新模型进入完整 5-fold 必须满足：

#### 基本通过

```text
screen_avg_BA >= final_screen_avg - 0.02
```

且：

```text
fold_4_BA >= final_fold_4 - 0.03
```

#### 强通过

```text
至少一个 screening fold 超过 final
且另一个不低于 final - 0.03
```

#### 直接淘汰

```text
screen_avg_BA < final_screen_avg - 0.05
或 fold_4_BA < final_fold_4 - 0.08
或 OOM 后降级仍失败
```

### 11.3 筛选输出

每个模型输出：

```text
outputs/long_model_arena/03_screening/{model_name}/screening_report.md
outputs/long_model_arena/03_screening/{model_name}/screening_metrics.csv
outputs/long_model_arena/03_screening/{model_name}/screening_predictions.csv
```

全局输出：

```text
outputs/long_model_arena/03_screening/model_screening_leaderboard.csv
```

---

## 12. 模型 A：TSception-lite

### 12.1 目标

测试多尺度 temporal convolution + spatial asymmetry 是否能在当前 raw EEG 任务中超过 EEGNet dual-branch。

### 12.2 输入

```text
raw EEG windows or trial10s
shape: [B, 30, T]
sampling rate: 250 Hz
```

优先使用和 exp3 相同窗口切法。

### 12.3 模型简化

实现 lite 版本：

```text
Temporal kernels:
- 0.5 s
- 1.0 s
- 2.0 s

Spatial branches:
- global spatial conv
- left/right asymmetric grouping if easy
```

推荐配置：

```yaml
num_T: 8
num_S: 8
hidden: 64
dropout: 0.5
batch_size: 16
epochs: 60
patience: 12
```

### 12.4 筛选

先跑：

```text
fold_1
fold_4
```

后处理：

```text
original_trial aggregation
subject top-4
```

如果通过筛选，跑 5-fold。

---

## 13. 模型 B：EEG-Conformer-lite

### 13.1 目标

测试 compact convolutional transformer 是否能比 EEGNet 更好捕捉局部+全局 EEG pattern。

### 13.2 输入

```text
raw EEG [B, 30, T]
```

### 13.3 Lite 配置

考虑 8GB 显存，使用小模型：

```yaml
emb_size: 32
depth: 2
num_heads: 4
patch_size: small
dropout: 0.5
batch_size: 8
mixed_precision: true
epochs: 60
patience: 12
```

如果 OOM：

```text
depth 2 -> 1
emb_size 32 -> 24
batch_size 8 -> 4
```

### 13.4 筛选与晋级

同 TSception。

---

## 14. 模型 C：LGGNet-lite

### 14.1 目标

测试 local-global-graph representation 是否能利用 30 通道脑区结构。

### 14.2 输入

优先使用：

```text
DE/PSD/FFT/STFT8 features
shape: [B, 30, 8, F]
```

如果已有 DE/PSD/STFT8 特征，复用；不要重新写复杂特征提取。

### 14.3 脑区划分

使用本地 30 通道划分：

```text
frontal:
FP1, FP2, F7, F3, FZ, F4, F8

fronto-central:
FT7, FC3, FCZ, FC4, FT8

central:
T3, C3, CZ, C4, T4

centro-parietal:
TP7, CP3, CPZ, CP4, TP8

parietal-occipital:
T5, P3, PZ, P4, T6, O1, OZ, O2
```

### 14.4 Lite 配置

```yaml
hidden_dim: 64
graph_layers: 1
dropout: 0.4
batch_size: 16
epochs: 60
patience: 12
```

### 14.5 风险

之前 STFT/GraphTransformer 表现不好。  
因此 LGGNet-lite 必须严格走 screening，不通过不跑完整 5-fold。

---

## 15. 模型 D：EEG-Deformer-lite / Dense Conv Transformer-lite

### 15.1 目标

测试 dense convolution + transformer 是否比普通 EEG-Conformer 稳。

### 15.2 输入

```text
raw EEG [B, 30, T]
```

### 15.3 Lite 配置

```yaml
conv_channels: 16 or 32
transformer_depth: 1
num_heads: 4
embedding_dim: 32
dropout: 0.4
batch_size: 8
epochs: 60
patience: 12
```

### 15.4 筛选

先 fold_1 / fold_4。

---

## 16. 模型 E：EmT-lite

### 16.1 目标

测试 temporal graph + transformer 的 cross-subject emotion recognition 思路。

### 16.2 输入

```text
x_feat: [B, 30, 8, F]
```

### 16.3 Lite 结构

简化为：

```text
per-time graph encoder
→ graph token sequence
→ 1-layer temporal transformer
→ classifier
```

配置：

```yaml
graph_hidden: 32
transformer_depth: 1
heads: 4
dropout: 0.4
batch_size: 8
epochs: 50
patience: 10
```

### 16.4 严格闸门

由于之前 GraphTransformer 失败，EmT-lite 只允许 screening。  
只有强通过才跑完整 5-fold。

---

## 17. 模型 F：DAMGCN-lite / Channel-Band Attention GCN

### 17.1 目标

测试通道图注意力 + 频带注意力是否能在 DE/PSD/FFT 特征上带来可解释提升。

### 17.2 输入

```text
x_feat: [B, 30, 8, F]
```

### 17.3 Lite 结构

```text
channel graph conv
band attention
channel attention
global pooling
classifier
```

配置：

```yaml
hidden: 64
graph_layers: 1
attention_heads: 2
dropout: 0.4
batch_size: 16
epochs: 60
patience: 12
```

### 17.4 筛选

先 fold_1 / fold_4。

---

## 18. 模型 G：exp3 small variants

这是最优先的新调优方向，因为它继承当前最强模型。

### 18.1 exp3 dropout variants

只试：

```text
dropout = original
dropout = original + 0.05
dropout = original - 0.05
```

先 fold_1 / fold_4。

### 18.2 exp3 weight decay variants

只试：

```text
weight_decay = 0.005
weight_decay = 0.01
weight_decay = 0.02
```

### 18.3 exp3 FFT variant

只试：

```text
current log_power
log1p_power
train_fold zscore_log_power
```

### 18.4 exp3 branch dropout

训练时随机 drop frequency branch 或 time branch，小概率：

```yaml
branch_dropout_p: 0.1
```

推理时正常双分支。

### 18.5 exp3 variant 晋级规则

任何 exp3 variant 只要 screening 接近 final，就优先于其它新模型跑 full 5-fold。

原因：

```text
当前最强 family 已经是 exp3
小改动风险最低
```

---

## 19. 训练统一配置

默认：

```yaml
epochs: 60
patience: 12
optimizer: AdamW
lr: 0.0005
weight_decay: 0.01
batch_size: 16
mixed_precision: true
gradient_clip_norm: 1.0
scheduler: cosine
monitor: val_balanced_accuracy
save_best: true
```

8GB safe：

```yaml
batch_size: 8
num_workers: 2
pin_memory: true
persistent_workers: false
```

OOM 自动降级：

```text
batch_size 16 -> 8 -> 4
model hidden dim reduce by half
retry once
then skip
```

---

## 20. 统一后处理

所有模型都必须通过统一后处理：

```text
1. 获取 window/trial probability
2. 聚合到 original_trial
3. subject top-4
4. original_trial metric
```

默认 aggregation：

```text
confidence_weighted_mean if window-level probabilities available
mean_prob fallback
```

如果模型输出 logits：

```text
softmax → positive class probability
```

如果二分类输出单 logit：

```text
sigmoid → positive probability
```

---

## 21. Ensemble 规则

只允许低风险 ensemble：

### 21.1 final + candidate 小权重 ensemble

如果某 candidate 在某些 fold 有互补性，可以试：

```text
score = w * final_score + (1 - w) * candidate_score
```

但只允许：

```text
w ∈ {0.8, 0.85, 0.9, 0.95}
```

也就是 final 权重大。

禁止 candidate 主导。

### 21.2 多 seed ensemble

```text
score = mean(seed_scores)
```

或：

```text
rank average
```

### 21.3 Promote 规则

ensemble 必须：

```text
BA > 0.7583
至少 3/5 folds >= final
std not worse
no public tuning
```

否则只作为 supplement。

---

## 22. 候选选择

生成：

```text
outputs/long_model_arena/06_candidate_selection/all_candidates.csv
outputs/long_model_arena/06_candidate_selection/candidate_decision.md
```

字段：

```csv
candidate,model_family,screening_or_full,ba,mf1,std,folds_ge_final,public_submission,decision,reason
```

决策：

```text
promote
supplement
reject
incomplete
invalid
```

---

## 23. 最终提交审计

最终候选必须生成：

```text
outputs/long_model_arena/07_submissions/final_submission_selected.xlsx
outputs/long_model_arena/07_submissions/final_submission_selected_with_prob.xlsx
outputs/long_model_arena/07_submissions/final_submission_audit.md
outputs/long_model_arena/07_submissions/final_submission_check.csv
```

检查：

```text
80 rows
10 users
8 trials/user
4 positive/user
4 neutral/user
no duplicates
no missing
0/1 labels
correct columns
xlsx readable
```

---

## 24. 最终报告

生成：

```text
outputs/long_model_arena/08_reports/long_model_arena_report.md
outputs/long_model_arena/08_reports/final_recommendation.md
outputs/long_model_arena/08_reports/team_update.md
```

### 24.1 long_model_arena_report.md

结构：

```markdown
# Long Model Arena Report

## 1. Safety baseline

## 2. Deterministic rebuild

## 3. Seed ensemble

## 4. Model candidates screened

| Model | Screening BA | Full BA | Decision |

## 5. New models

### TSception-lite
### EEG-Conformer-lite
### LGGNet-lite
### EEG-Deformer-lite
### EmT-lite
### DAMGCN-lite
### exp3 variants

## 6. Ensembles

## 7. Final selected candidate

## 8. Why old failed routes remain stopped

## 9. Risks

## 10. Recommendation
```

### 24.2 final_recommendation.md

必须回答：

```text
最终应该提交哪个文件？
为什么？
相比 frozen 有没有提升？
如果没有提升，为什么仍然提交 frozen？
备份文件是什么？
```

### 24.3 team_update.md

写给队友：

```markdown
# 队友更新

我们这次没有只跑报告，而是实际筛选了多个新模型：

- TSception-lite
- EEG-Conformer-lite
- LGGNet-lite
- EEG-Deformer-lite
- EmT-lite
- DAMGCN-lite
- exp3 variants
- seed ensemble

最终结果：
...

建议：
...
```

---

## 25. 进度日志

每个模型开始和结束都要写：

```text
outputs/long_model_arena/run_state/progress.md
```

格式：

```markdown
## YYYY-MM-DD HH:MM - Started TSception-lite screening

Folds:
- fold_1
- fold_4

Config:
...

## YYYY-MM-DD HH:MM - Finished TSception-lite screening

Results:
- fold_1 BA:
- fold_4 BA:
- decision:
- next:
```

如果任务中断，重启后先读 progress.md。

---

## 26. 失败处理

### 26.1 OOM

自动：

```text
batch_size halve
hidden dim reduce
retry once
```

若仍失败：

```text
mark reject_oom
continue next model
```

### 26.2 代码适配失败

如果一个新模型 30 分钟内适配不通：

```text
mark adapter_failed
continue next model
```

不要卡死 7 小时。

### 26.3 训练过慢

如果某模型单 fold 预计超过 90 分钟：

```text
stop after current epoch
mark too_slow
continue next
```

### 26.4 筛选失败

如果 screening 明显低于 final：

```text
do not run full 5-fold
continue next model
```

### 26.5 单 fold 离群提升

如果只有 fold_1 大幅提升：

```text
mark supplement_outlier
do not promote
```

---

## 27. 最终终端摘要

任务结束后打印：

```text
===== Long Model Arena Summary =====

Safety baseline:
  exp3_vote_alpha062_cwmean_top4
  BA:
  MF1:
  submission:

Deterministic rebuild:
  verdict:
  BA:
  same as frozen:

Seed ensemble:
  seeds complete:
  best ensemble:
  BA:
  MF1:
  decision:

New model screening:
  TSception-lite:
  EEG-Conformer-lite:
  LGGNet-lite:
  EEG-Deformer-lite:
  EmT-lite:
  DAMGCN-lite:
  exp3 variants:

Full 5-fold promoted candidates:
  ...

Best candidate:
  name:
  BA:
  MF1:
  folds >= final:
  decision:

Selected submission:
  path:
  SHA256:
  audit:

Recommendation:
  SUBMIT_FROZEN / SUBMIT_NEW_CANDIDATE / SUBMIT_ENSEMBLE

Do not continue:
  AdaBN / DANN / external pretraining / large contrastive / Graph routes that failed screening

==================================
```

---

## 28. 最终执行原则

这份计划允许新模型探索，但必须始终记住：

```text
当前 final 是安全底线。
新模型必须用统一 top-4 口径证明自己。
只提升单 fold 不算成功。
不能用 public 调参。
不能为了跑新模型破坏已锁定提交。
```

如果所有新模型都失败，最终推荐仍然是：

```text
outputs/final_review/final_submission_clean.xlsx
```

如果新模型或 ensemble 稳定超过 final，才生成新的 selected submission。
