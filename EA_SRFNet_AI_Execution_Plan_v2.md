# EA-SRFNet / Conformer+SRFNet 后续实验 AI 执行计划

> 项目：BCIMI / EEG Emotion Recognition  
> 目标：在不破坏当前最优主线的前提下，验证 Euclidean Alignment（EA）是否能进一步提升跨被试 EEG 情绪识别稳定性。  
> 当前基线：`Conformer + SRFNet subject-level rank ensemble`，`w_conformer=0.55`，`w_srfnet=0.45`，CV BA = `0.7958`。  
> 核心原则：**先冻结主线，再做 EA 副线；EA 只有完整 5-fold + 审计通过 + 超过当前主线，才允许升级为新主提交。**

---

## 0. 项目背景摘要

当前项目的关键认知：

```text
本任务不是普通 trial-level threshold 二分类，
而是 subject 内 8 个 trial 的相对排序问题。
```

每名测试 subject：

```text
8 个 trial = 4 个 positive + 4 个 neutral
```

因此当前主线使用：

```text
每 subject 内 original_trial score 排序
score top 4 → Emotion_label = 1
score bottom 4 → Emotion_label = 0
```

当前最优：

```text
Conformer + SRFNet rank ensemble
w_conformer = 0.55
w_srfnet = 0.45
w_exp3 = 0.00
CV BA = 0.7958
```

SRFNet 单模型：

```text
SRFNet trial0007_long_e6e10
CV BA = 0.7833
Per-fold = [0.7500, 0.8333, 0.7917, 0.7917, 0.7500]
Decision = SUPPLEMENT_STRONG
```

采样率审计：

```text
SFREQ = 250 Hz
Window = 4.0s
n_times = 1000
128Hz 是旧文档 typo
Decision = PASS
```

---

## 1. 为什么现在可以尝试 EA

EA，即 Euclidean Alignment，核心作用是对每个 subject 的 EEG 协方差进行白化/对齐，减少跨被试空间协方差差异。

本赛题本质是 cross-subject EEG emotion recognition，subject shift 是主要困难之一。EA 可能有帮助的原因：

```text
1. 训练 subject 和测试 subject 不同；
2. EEG 个体差异明显；
3. 当前模型仍依赖 raw EEG 空间结构；
4. SRFNet 的 RawConformerBranch 对通道协方差和空间分布敏感；
5. EA 是无监督方法，可以在不使用 label 的情况下对 public/private subject 做 subject-level normalization。
```

但 EA 也可能有害：

```text
1. 情绪相关信息可能部分存在于功率/协方差模式中；
2. EA 全输入可能破坏 FFT/log_power 分支的频域物理意义；
3. 之前 AdaBN 类对齐方法曾经负收益；
4. 当前最优已经较强，EA 只能作为副线候选，不能直接替换主线。
```

因此本计划优先测试：

```text
subject_self + raw_branch_only EA
```

也就是：

```text
只让 SRFNet 的 RawConformerBranch 吃 EA 后 EEG；
FFT branch / regional branch 仍使用原始 z-score EEG。
```

---

## 2. 总目标

实现一个可开关的 EA 预处理模块，并严格比较：

```text
A. 原始 SRFNet
B. EA-SRFNet all_input
C. EA-SRFNet raw_branch_only
D. 原始 Conformer + 原始 SRFNet ensemble
E. Conformer + EA-SRFNet ensemble
F. Conformer + 原始 SRFNet + EA-SRFNet ensemble
G. 当前最优模型去掉 subject top-4 后处理，再叠加 EA，观察 trial-level 判别/校准能力
```

最终只允许四类结论：

```text
1. PROMOTE_EA_ENSEMBLE
2. KEEP_CURRENT_CONFORMER_SRFNET_ENSEMBLE
3. USE_EA_SRFNET_AS_SUPPLEMENT_ONLY
4. REJECT_EA
```

---

## 3. 最高优先级约束

### 3.1 禁止事项

禁止：

```text
1. 禁止覆盖当前最优 Conformer+SRFNet 提交文件；
2. 禁止覆盖已有 Conformer / SRFNet / exp3 / frozen baseline 结果；
3. 禁止使用 public label；
4. 禁止使用 public 反馈调参；
5. 禁止 per-fold oracle 权重；
6. 禁止 per-subject 权重；
7. 禁止 stacking / meta-classifier；
8. 禁止修改 top-4 规则；
9. 禁止只凭单 fold 结果宣布提升；
10. 禁止没有完整 5-fold 就纳入最终主提交；
11. 禁止删除、移动、覆盖旧实验结果；
12. 禁止把 EA 和 AdaBN/DANN 混在一起做大网格搜索；
13. 禁止 private/public 数据标签泄露；
14. 禁止在主提交/主结论中用 threshold=0.5 替代 subject top-4；Phase 8B 的 no-top4 只作为消融分析，不参与 submission/promote。
```

### 3.2 允许事项

允许：

```text
1. 新建 EA 输出目录；
2. 新增 EA 预处理代码；
3. 使用无标签 EEG 信号估计每个 subject 的 EA 矩阵；
4. 在 train/val/test/public 各自 subject 内估计 EA；
5. 做原始 SRFNet 与 EA-SRFNet 的固定权重 rank ensemble；
6. 做 Conformer + EA-SRFNet 固定权重 rank ensemble；
7. 做完整 5-fold 复算；
8. 生成新的候选 submission；
9. 做 submission audit 和 SHA256 hash；
10. 生成最终报告。
```

---

## 4. 输出目录

所有 EA 新实验统一写入：

```text
outputs/ea_srfnet_experiment/
```

建议目录：

```text
outputs/ea_srfnet_experiment/
├── 00_source_inventory/
├── 01_ea_implementation_audit/
├── 02_ea_srfnet_smoke/
├── 03_ea_srfnet_5fold/
├── 04_score_alignment/
├── 05_ensemble_with_ea/
├── 05b_no_top4_ea_ablation/
├── 06_submission_candidates/
├── 07_final_audit/
├── 08_reports/
└── run_state/
```

每个 Phase 更新：

```text
outputs/ea_srfnet_experiment/run_state/progress.md
```

---

## 5. Phase 0：Source Inventory

### 5.1 目标

确认当前所有必要输入结果都存在，避免 EA 实验建立在错误 score source 上。

### 5.2 必须搜索路径

```text
outputs/srfnet/
auto_tune_runs/srfnet_first_live/
outputs/srfnet_three_model_final/
outputs/final_submission_primary/
outputs/conformer_exp3_ensemble/
outputs/eeg_conformer_tuning/
outputs/final_review/
EEGNET/outputs/
models/
training/
```

### 5.3 必须登记文件

生成：

```text
outputs/ea_srfnet_experiment/00_source_inventory/source_inventory.csv
```

字段：

```csv
name,path,exists,file_type,model,version,contains_cv_scores,contains_public_scores,contains_submission,notes
```

必须包含：

```text
1. SRFNet model file: models/srfnet.py 或等价路径；
2. SRFNet train script: train_srfnet.py 或等价路径；
3. trial0007_long_e6e10 config；
4. trial0007_long_e6e10 original_trial_scores.csv；
5. trial0007_long_e6e10 top4_predictions_cv.csv；
6. trial0007_long_e6e10 summary_metrics.json；
7. Conformer original_trial_scores.csv；
8. Conformer public original_trial_scores.csv；
9. Conformer+SRFNet best ensemble scores；
10. Conformer+SRFNet best submission；
11. public template xlsx；
12. manifest files；
13. split_manifest.json。
```

### 5.4 输出

```text
outputs/ea_srfnet_experiment/00_source_inventory/source_inventory.csv
outputs/ea_srfnet_experiment/00_source_inventory/source_inventory_report.md
```

---

## 6. Phase 1：EA 模块实现

### 6.1 新增文件

新增：

```text
EEGNET/data/euclidean_alignment.py
```

实现函数：

```python
def compute_covariance(x, eps=1e-5):
    """
    x: Tensor or ndarray, shape [channels, times]
    return: normalized covariance, shape [channels, channels]
    """


def compute_subject_reference_covariance(windows, eps=1e-5):
    """
    windows: [N, channels, times]
    return: mean normalized covariance R_s
    """


def inv_sqrtm_spd(cov, eps=1e-5):
    """
    cov: SPD matrix [channels, channels]
    return: cov^{-1/2}
    """


def fit_ea_transform(windows, eps=1e-5):
    """
    windows: [N, channels, times]
    return: transform matrix R_s^{-1/2}, diagnostics dict
    """


def apply_ea_transform(x, transform):
    """
    x: [channels, times] or [N, channels, times]
    transform: [channels, channels]
    return: aligned x
    """
```

### 6.2 EA 公式

对 subject `s` 的所有 windows：

```text
C_i = X_i X_i^T / trace(X_i X_i^T)
R_s = mean_i(C_i)
X_i_aligned = R_s^{-1/2} X_i
```

要求：

```text
1. 每个 subject 单独估计 R_s；
2. 只使用 EEG 信号，不使用 label；
3. 对协方差加 eps * I；
4. trace 太小时做保护；
5. R_s 必须强制对称化：R_s = (R_s + R_s.T) / 2；
6. R_s^{-1/2} 用 eigh 实现；
7. 保存每个 subject 的诊断信息；
8. 不允许 silently ignore NaN / Inf。
```

### 6.3 伪代码

```python
import numpy as np


def compute_covariance(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    cov = x @ x.T
    tr = np.trace(cov)
    if not np.isfinite(tr) or tr <= eps:
        tr = eps
    cov = cov / tr
    cov = (cov + cov.T) / 2.0
    cov = cov + eps * np.eye(cov.shape[0], dtype=np.float64)
    return cov


def fit_ea_transform(windows, eps=1e-5):
    covs = [compute_covariance(w, eps=eps) for w in windows]
    ref = np.mean(covs, axis=0)
    ref = (ref + ref.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(ref)
    eigvals = np.maximum(eigvals, eps)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    inv_sqrt = (inv_sqrt + inv_sqrt.T) / 2.0
    diagnostics = {
        "eig_min": float(eigvals.min()),
        "eig_max": float(eigvals.max()),
        "condition_number": float(eigvals.max() / eigvals.min()),
        "eps": eps,
    }
    return inv_sqrt.astype(np.float32), diagnostics
```

---

## 7. Phase 2：EA 接入 Dataset / Training

### 7.1 参数设计

在 SRFNet 训练脚本中新增参数：

```text
--ea-mode
choices = ["none", "subject_self", "train_subject_reference"]
default = "none"

--ea-target
choices = ["all_input", "raw_branch_only"]
default = "all_input"

--ea-eps
default = 1e-5

--cache-ea
action = store_true
```

### 7.2 ea-mode 语义

#### none

```text
不使用 EA。
```

#### subject_self

每个 subject 用自己的全部无标签 windows 估计 EA。

CV 内部：

```text
train subject: 用该 train subject 的所有 train windows 估计 EA；
val subject: 用该 val subject 的所有 val windows 估计 EA；
test subject: 用该 test subject 的所有 test windows 估计 EA。
```

public/private 推理：

```text
每个 public/private subject 用自己的 8 trials / all windows 估计 EA。
```

注意：

```text
这是 unsupervised transductive subject normalization；
不使用 label；
不使用 public feedback；
必须在报告中显式说明。
```

#### train_subject_reference

只使用训练 subjects 估计一个 global train reference covariance，然后应用到 val/test/public。

用途：

```text
作为更保守的对照版本。
```

### 7.3 ea-target 语义

#### all_input

```text
SRFNet 所有分支都吃 EA 后 EEG。
```

#### raw_branch_only

```text
RawConformerBranch 吃 EA 后 EEG；
FftEEGNetBranch 仍吃原始 z-score EEG；
RegionalSummaryEncoder 仍吃原始 z-score EEG。
```

推荐优先级：

```text
1. subject_self + raw_branch_only
2. subject_self + all_input
3. train_subject_reference + raw_branch_only
4. train_subject_reference + all_input
```

---

## 8. Phase 3：SRFNet 结构适配 raw_branch_only

### 8.1 forward 接口

修改 SRFNet：

```python
def forward_features(self, x, x_ea=None):
    if x_ea is None:
        x_raw = x
    else:
        x_raw = x_ea

    z_conf, score_conf = self.raw_branch(x_raw)
    z_fft, score_fft = self.fft_branch(x)

    if self.region_branch is not None:
        z_region = self.region_branch(x)

    ...
```

保持兼容：

```python
def forward(self, x, x_ea=None):
    outputs = self.forward_features(x, x_ea=x_ea)
    score = outputs["score"]
    outputs["logits"] = torch.stack([-score, score], dim=1)
    return outputs
```

### 8.2 DataLoader batch 字段

当 `ea-target=raw_branch_only` 时，batch 应提供：

```python
batch["features"]      # 原始预处理 EEG
batch["features_ea"]   # EA 后 EEG
```

训练逻辑：

```python
if args.ea_mode != "none" and args.ea_target == "raw_branch_only":
    outputs = model(features, x_ea=features_ea)
elif args.ea_mode != "none" and args.ea_target == "all_input":
    outputs = model(features_ea)
else:
    outputs = model(features)
```

### 8.3 兼容性要求

```text
1. 原有 checkpoint 可正常加载；
2. --ea-mode none 时输出应与旧代码一致；
3. 不改变现有 top-4 评估逻辑；
4. 不改变原有 loss 默认配置；
5. 不影响 public submission 写出格式。
```

---

## 9. Phase 4：EA 实现审计

### 9.1 诊断文件

生成：

```text
outputs/ea_srfnet_experiment/01_ea_implementation_audit/ea_subject_stats.csv
```

字段：

```csv
split,fold,subject_id,n_windows,n_channels,n_times,trace_mean,eig_min,eig_max,condition_number,eps,status,notes
```

### 9.2 异常条件

标记 WARNING：

```text
eig_min <= 0
condition_number > 1e6
NaN / Inf
n_windows < 8
shape != [N, 30, 1000]
```

如果异常，按以下 eps 重试：

```text
eps ∈ {1e-5, 1e-4, 1e-3}
```

如果仍失败：

```text
该 subject 回退到 identity transform；
记录为 FALLBACK_IDENTITY；
不允许静默跳过。
```

### 9.3 输出

```text
outputs/ea_srfnet_experiment/01_ea_implementation_audit/ea_implementation_audit.md
outputs/ea_srfnet_experiment/01_ea_implementation_audit/ea_subject_stats.csv
```

---

## 10. Phase 5：fold_0 smoke test

### 10.1 目标

只跑 fold_0，低成本排除明显负收益 EA。

### 10.2 实验矩阵

基于当前 SRFNet trial0007_long_e6e10 配置，跑：

```text
S0: baseline reproduce
    ea_mode = none

S1: subject_self + all_input
    ea_mode = subject_self
    ea_target = all_input

S2: subject_self + raw_branch_only
    ea_mode = subject_self
    ea_target = raw_branch_only

S3: train_subject_reference + all_input
    ea_mode = train_subject_reference
    ea_target = all_input

S4: train_subject_reference + raw_branch_only
    ea_mode = train_subject_reference
    ea_target = raw_branch_only
```

### 10.3 输出目录

```text
outputs/ea_srfnet_experiment/02_ea_srfnet_smoke/
├── S0_baseline_reproduce/
├── S1_subject_self_all_input/
├── S2_subject_self_raw_only/
├── S3_train_ref_all_input/
└── S4_train_ref_raw_only/
```

### 10.4 每个 smoke 输出

```text
fold_metrics.csv
top4_predictions.csv
original_trial_scores.csv
summary_metrics.json
ea_diagnostics.csv
training_history.csv
smoke_report.md
```

### 10.5 smoke 晋级规则

只允许满足以下条件的版本进入 5-fold：

```text
fold_0 BA >= baseline_fold0_BA - 0.0200
训练无 NaN / Inf
top-4 audit PASS
EA diagnostics 无严重错误
```

如果多个通过，最多选 2 个进入 5-fold。

优先级：

```text
1. S2 subject_self + raw_branch_only
2. S1 subject_self + all_input
3. S4 train_subject_reference + raw_branch_only
4. S3 train_subject_reference + all_input
```

---

## 11. Phase 6：EA-SRFNet 完整 5-fold

### 11.1 目标

对 smoke 通过的 EA 版本跑完整 5-fold。

### 11.2 最多两个版本

```text
EA_MAIN = smoke 最强，优先 subject_self + raw_branch_only
EA_ALT = 机制不同且 smoke 合格的次强版本
```

建议默认：

```text
EA_MAIN = subject_self + raw_branch_only
EA_ALT = subject_self + all_input
```

### 11.3 必须复算指标

```text
Balanced Accuracy
Macro-F1
ROC-AUC
Accuracy
Precision
Recall
Per-fold BA
BA std
min_fold_BA
fold3_BA
fold4_BA
confusion matrix
per-subject accuracy
```

### 11.4 输出

```text
outputs/ea_srfnet_experiment/03_ea_srfnet_5fold/
├── EA_MAIN/
│   ├── fold_metrics.csv
│   ├── top4_predictions_cv.csv
│   ├── original_trial_scores.csv
│   ├── summary_metrics.json
│   ├── ea_diagnostics.csv
│   └── report.md
└── EA_ALT/
    ├── fold_metrics.csv
    ├── top4_predictions_cv.csv
    ├── original_trial_scores.csv
    ├── summary_metrics.json
    ├── ea_diagnostics.csv
    └── report.md
```

### 11.5 单模判定

EA-SRFNet 单模判定：

```text
如果 BA > 0.7833 且 5-fold 完整：EA_SINGLE_IMPROVED
如果 BA ≈ 0.7833 但 std 更低：EA_SINGLE_STABILITY_CANDIDATE
如果 BA < 0.7833：EA_SINGLE_NO_GAIN
如果 BA < 0.7750：EA_SINGLE_REJECT
```

---

## 12. Phase 7：Score Alignment

### 12.1 目标

对齐以下模型 original_trial scores：

```text
1. tuned Conformer
2. original SRFNet long
3. EA-SRFNet MAIN
4. EA-SRFNet ALT，如果存在
5. current Conformer+SRFNet ensemble
```

### 12.2 对齐 key

```text
fold
subject_id
original_trial_id
y_true
```

### 12.3 检查

```text
1. 每个模型覆盖相同 folds；
2. 每 fold 覆盖相同 subjects；
3. 每 subject 恰好 8 个 original_trial；
4. y_true 完全一致；
5. original_trial_id 完全一致；
6. 无重复 key；
7. 无缺失 key；
8. 每个模型单独 top-4 后能复现报告 BA。
```

### 12.4 输出

```text
outputs/ea_srfnet_experiment/04_score_alignment/
├── aligned_scores_long.csv
├── score_alignment_report.md
├── missing_or_duplicate_keys.csv
├── model_score_reproduction.csv
└── per_model_top4_recomputed.csv
```

---

## 13. Phase 8：Conformer + EA-SRFNet Ensemble

### 13.1 目标

判断 EA 是否能超过当前最优：

```text
Current best:
Conformer + original SRFNet
w_conformer = 0.55
w_srfnet = 0.45
BA = 0.7958
```

### 13.2 Ensemble 方法

全部使用 original_trial 层面的 rank ensemble。

每 subject 内：

```text
rank_model = rank(score_model among 8 trials)
rank_score = weighted average rank
rank_score top 4 → positive
```

注意统一方向：

```text
分数越大越 positive；
rank 越靠前越 positive；
实现时必须明确 ascending / descending。
```

### 13.3 候选

```text
E0: current Conformer + original SRFNet
E1: Conformer + EA-SRFNet_MAIN
E2: Conformer + original SRFNet + EA-SRFNet_MAIN
E3: original SRFNet + EA-SRFNet_MAIN
E4: Conformer + EA-SRFNet_ALT，如果 ALT 存在
E5: Conformer + original SRFNet + EA-SRFNet_ALT，如果 ALT 存在
```

### 13.4 权重搜索

Two-model：

```text
w ∈ {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}
```

Three-model：

```text
w_conformer ∈ {0.40, 0.45, 0.50, 0.55, 0.60}
w_srfnet    ∈ {0.15, 0.20, 0.25, 0.30, 0.35}
w_ea        ∈ {0.15, 0.20, 0.25, 0.30, 0.35}
保留 sum = 1.0 ± 1e-6
```

禁止：

```text
per-fold weight
per-subject weight
threshold tuning
public feedback
stacking
```

### 13.5 Promote 标准

只有满足以下条件，EA ensemble 才能替代当前最优：

```text
BA > 0.7958
Macro-F1 >= 当前最优 Macro-F1
min_fold_BA 不下降超过 0.02
fold3_BA 不下降
fold4_BA 不下降
std 不明显增加
public 未参与选择
submission audit PASS
全局固定权重
```

如果：

```text
BA = 0.7958
但 std 更低 / min_fold 更高 / fold3 更稳
```

则标记：

```text
PRIMARY_STABILITY_CANDIDATE
```

但不自动替换主提交。

### 13.6 输出

```text
outputs/ea_srfnet_experiment/05_ensemble_with_ea/
├── ea_ensemble_leaderboard.csv
├── ea_ensemble_per_fold_results.csv
├── stable_objective_leaderboard.csv
├── best_ea_ensemble_top4_predictions.csv
├── best_ea_ensemble_original_trial_scores.csv
└── ea_ensemble_report.md
```

---

## 13B. Phase 8B：当前最优模型去 top-4 + EA 消融路线

### 13B.1 目标

新增一条独立路线：

```text
当前最优 Conformer+SRFNet 模型
- 去掉 subject top-4 结构化解码
+ 加入 EA 版本的 SRFNet / EA 输入
→ 观察 trial-level 判别能力和概率校准能力
```

这条路线的核心问题不是“能否生成最终提交”，而是回答：

```text
如果不使用每 subject 必有 4 positive / 4 neutral 的结构先验，
EA 是否仍能提升模型对单个 original_trial 的直接分类能力？
```

### 13B.2 重要定位

这是一条 **消融/诊断路线**，不是默认提交路线。

原因：

```text
1. 赛题数据结构明确每个测试 subject 为 4 positive + 4 neutral；
2. 当前主线的核心优势正是 subject-level ranking + top-4 decoding；
3. 去 top-4 会主动放弃强先验，预期 BA 可能显著下降；
4. 但该路线能判断 EA 是否改善了模型本身的 raw score / probability quality。
```

因此：

```text
NoTop4 结果只进入 ablation report；
不参与最终 submission 候选；
不允许因为 NoTop4 单项变好而替换主提交；
不允许用 public feedback 调阈值。
```

### 13B.3 对照对象

至少比较以下对象：

```text
N0: CurrentBest_Top4
    Conformer + original SRFNet rank ensemble
    w_cf = 0.55, w_srf = 0.45
    decoding = subject_top4

N1: CurrentBest_NoTop4_FixedProb
    Conformer + original SRFNet score/prob ensemble
    w_cf = 0.55, w_srf = 0.45
    decoding = fixed threshold 0.5

N2: Conformer_EA-SRFNet_NoTop4_FixedProb
    Conformer + EA-SRFNet score/prob ensemble
    w_cf = 0.55, w_ea = 0.45
    decoding = fixed threshold 0.5

N3: Conformer_SRFNet_EA-SRFNet_NoTop4_FixedProb
    Conformer + original SRFNet + EA-SRFNet score/prob ensemble
    使用 Phase 8 中最优的全局固定权重，或默认 0.50/0.25/0.25
    decoding = fixed threshold 0.5

N4: EA-SRFNet_Single_NoTop4_FixedProb
    EA-SRFNet 单模型
    decoding = fixed threshold 0.5
```

如果原始 Conformer score 没有概率校准，只能拿 rank score，则额外记录：

```text
rank_score 不是概率；
fixed threshold 不可解释；
该项只做排序/分布诊断，不做正式 NoTop4 BA 结论。
```

### 13B.4 可选阈值诊断，但禁止作为主结论

为了了解校准上限，可以额外做一个诊断项：

```text
N*_NoTop4_ValCalibratedThreshold
```

规则：

```text
1. 每个 fold 的 threshold 只能由该 fold 的 validation set 决定；
2. 不允许用 test fold y_true 选 threshold；
3. 不允许用 public feedback 选 threshold；
4. 报告中必须标记为 calibration diagnostic；
5. 不进入最终主候选。
```

主比较仍以：

```text
fixed threshold = 0.5
```

为准。

### 13B.5 指标

对 NoTop4 路线计算：

```text
Balanced Accuracy
Macro-F1
Accuracy
Precision
Recall
ROC-AUC
Brier score
ECE，如项目已有校准函数则计算
Per-fold BA
Per-subject positive count distribution
```

重点检查：

```text
1. 每个 subject 被预测为 positive 的数量是否接近 4；
2. 是否出现全 0 / 全 1 subject；
3. EA 是否让 probability 更分散或更塌缩；
4. EA 是否提升 AUC 但不提升 BA；
5. EA 是否提升 calibration，但不提升 top-4 排序。
```

### 13B.6 输出文件

输出到：

```text
outputs/ea_srfnet_experiment/05b_no_top4_ea_ablation/
```

文件：

```text
no_top4_leaderboard.csv
no_top4_per_fold_metrics.csv
no_top4_per_subject_positive_count.csv
no_top4_probability_distribution.csv
no_top4_threshold_diagnostics.csv
no_top4_vs_top4_comparison.csv
no_top4_ea_ablation_report.md
```

### 13B.7 判断标准

NoTop4 路线的合理结论只有以下几类：

```text
A. EA improves raw trial-level classification
   EA-NoTop4 BA / MF1 / AUC 均优于 NoEA-NoTop4；
   说明 EA 改善了模型本身判别能力。

B. EA improves ranking but not calibration
   EA-Top4 提升，但 EA-NoTop4 不提升；
   说明 EA 主要改善 subject 内相对排序，而非全局概率阈值。

C. top-4 prior dominates
   NoTop4 全部显著低于 Top4；
   说明本赛题应继续坚持 subject-level top-4。

D. EA hurts raw score quality
   EA-NoTop4 和 EA-Top4 都下降；
   说明 EA 不适合当前主线。
```

### 13B.8 禁止事项

```text
1. 禁止把 NoTop4 public submission 作为正式提交；
2. 禁止用 public 准确率选择 NoTop4 threshold；
3. 禁止把 NoTop4 的阈值调优结果和 Top4 主线直接公平比较；
4. 禁止因为 NoTop4 某个 fold 变好就宣布 EA 提升；
5. 禁止覆盖当前 Top4 主线结果。
```

### 13B.9 报告里必须写清楚的一句话

```text
NoTop4 + EA 是诊断模型原始判别/校准能力的消融实验；
它不改变本项目最终应使用 subject-level top-4 decoding 的主策略。
```

---

## 14. Phase 9：Public Submission 生成与审计

### 14.1 候选 submission

至少生成：

```text
C0_current_conformer_srfnet.xlsx
C1_ea_srfnet_single_main.xlsx
C2_conformer_ea_srfnet_main.xlsx
C3_conformer_srfnet_ea_srfnet_main.xlsx
```

如果 EA_ALT 完整 5-fold 并进入 ensemble，则额外生成：

```text
C4_conformer_ea_srfnet_alt.xlsx
C5_conformer_srfnet_ea_srfnet_alt.xlsx
```

### 14.2 submission 规则

```text
每 user 8 trials
score top 4 → Emotion_label = 1
score bottom 4 → Emotion_label = 0
```

### 14.3 审计要求

每个文件检查：

```text
80 rows
10 users
8 trials/user
4 positive/user
4 neutral/user
no duplicated user_id + trial_id
Emotion_label ∈ {0,1}
columns match template
Excel readable
clean submission 不含 prob 等多余列
with_prob 版本可额外保存
SHA256 generated
```

### 14.4 输出

```text
outputs/ea_srfnet_experiment/06_submission_candidates/
├── submission_candidate_list.csv
├── submission_audit_all.md
├── C0_current_conformer_srfnet.xlsx
├── C1_ea_srfnet_single_main.xlsx
├── C2_conformer_ea_srfnet_main.xlsx
├── C3_conformer_srfnet_ea_srfnet_main.xlsx
├── *_with_prob.xlsx
└── hashes.txt
```

---

## 15. Phase 10：最终候选榜与决策

### 15.1 输出文件

生成：

```text
outputs/ea_srfnet_experiment/07_final_audit/final_candidate_board.csv
```

字段：

```csv
candidate,model_family,ea_mode,ea_target,ba,mf1,auc,ba_std,fold0,fold1,fold2,fold3,fold4,min_fold_ba,beats_current_07958,submission_path,sha256,decision,reason
```

### 15.2 决策规则

#### PRIMARY

```text
BA > 0.7958
MF1 >= 当前最优
submission audit PASS
```

#### PRIMARY_STABILITY

```text
BA = 0.7958
但 min_fold_BA / std / fold3 明显更好
submission audit PASS
```

#### SUPPLEMENT_STRONG

```text
0.7833 <= BA < 0.7958
或 EA 单模提升明显但 ensemble 未超过当前最优
```

#### BACKUP

```text
完整 5-fold，但低于当前主线；
可作为答辩补充或备份。
```

#### REJECT

```text
BA < 原 SRFNet
或 fold 波动变大
或破坏 top4
或审计不通过
或无法复现
```

### 15.3 最终主线保护

如果 EA 没有严格超过当前最优：

```text
final selected 仍为 Conformer + SRFNet rank ensemble w=0.55/0.45
```

不要因为 EA 是新方法就替换主提交。

---

## 16. Phase 11：最终报告

生成：

```text
outputs/ea_srfnet_experiment/08_reports/
├── ea_srfnet_final_report.md
├── ea_implementation_audit.md
├── ea_vs_noea_comparison.md
├── final_recommendation.md
├── team_update.md
└── technical_summary_for_ppt.md
```

### 16.1 ea_srfnet_final_report.md 结构

```markdown
# EA-SRFNet Final Report

## 1. Motivation

说明为什么尝试 EA：
- EEG 跨被试差异大；
- EA 可进行 subject-level covariance alignment；
- 当前模型强，但仍可能受 subject shift 影响；
- EA 作为副线候选，不直接覆盖主线。

## 2. Method

说明：
- EA covariance 计算；
- subject_self；
- train_subject_reference；
- all_input；
- raw_branch_only；
- 没有使用 label；
- 没有使用 public feedback。

## 3. Implementation Audit

展示：
- EA diagnostics；
- eig_min / eig_max；
- condition number；
- fallback 情况；
- shape check。

## 4. Smoke Test

展示 fold_0 结果：
- S0 baseline；
- S1 subject_self_all_input；
- S2 subject_self_raw_only；
- S3 train_ref_all_input；
- S4 train_ref_raw_only；
- 晋级版本。

## 5. 5-Fold EA Results

展示 EA_MAIN / EA_ALT：
- BA；
- MF1；
- AUC；
- per-fold；
- std；
- min_fold；
- fold3/fold4。

## 6. Ensemble Results

比较：
- current Conformer+SRFNet；
- Conformer+EA-SRFNet；
- Conformer+SRFNet+EA-SRFNet。

## 7. NoTop4 + EA Ablation

说明：
- 去掉 subject top-4 后处理后的 trial-level fixed-threshold 表现；
- EA 是否改善 raw probability / calibration；
- 每 subject positive count 是否偏离 4；
- 该路线只作为消融诊断，不参与最终提交。

## 8. Submission Audit

列出所有候选文件和 SHA256。

## 9. Final Decision

只能写以下之一：
- PROMOTE_EA_ENSEMBLE
- KEEP_CURRENT_CONFORMER_SRFNET_ENSEMBLE
- USE_EA_SRFNET_AS_SUPPLEMENT_ONLY
- REJECT_EA

## 10. Caveats

必须说明：
- EA 是无监督 test-time subject normalization；
- CV 提升不保证 public/private 提升；
- top-4 先验来自赛题结构；
- 不用 public label；
- 不用 public feedback 调参。
```

### 16.2 team_update.md 结构

```markdown
# 队内更新：EA-SRFNet 实验

## 当前主线

Conformer + SRFNet rank ensemble 仍是当前主线，BA=0.7958。

## 为什么尝试 EA

EA 试图减少跨被试 covariance shift，但不直接覆盖主线。

## 实验结果

...

## 最终推荐

...

## 提交文件

...
```

### 16.3 technical_summary_for_ppt.md 结构

```markdown
# EA-SRFNet 技术摘要

## 核心问题

跨被试 EEG 情绪识别存在 subject shift。

## EA 思路

对每个 subject 的 EEG 协方差做 Euclidean Alignment，使不同 subject 的空间协方差分布更一致。

## 与 SRFNet 结合

RawConformerBranch 使用 EA 后信号，FFT/Regional 分支保留原始 z-score 信号。

## 优势

1. 无监督；
2. 不使用测试标签；
3. 与 subject top-4 排序解码兼容；
4. 可作为跨被试泛化增强模块。

## 风险

1. 可能抹掉情绪相关协方差信息；
2. 全输入 EA 可能破坏频域功率特征；
3. 需要 5-fold 审计确认。
```

---

## 17. 最终终端摘要格式

任务结束后打印：

```text
===== EA-SRFNet Final Summary =====

Current best:
  model: Conformer + SRFNet rank ensemble
  weights: w_conformer=0.55, w_srfnet=0.45
  BA: 0.7958
  folds:

EA implementation:
  status:
  ea_mode tested:
  ea_target tested:
  leakage audit:

Smoke test:
  S0 baseline fold0:
  S1 subject_self_all_input fold0:
  S2 subject_self_raw_only fold0:
  S3 train_ref_all_input fold0:
  S4 train_ref_raw_only fold0:
  promoted variants:

5-fold EA:
  best EA single model:
  BA:
  MF1:
  AUC:
  folds:
  decision:

Ensemble with EA:
  best method:
  weights:
  BA:
  MF1:
  fold0:
  fold1:
  fold2:
  fold3:
  fold4:
  beats current 0.7958:

Final selected:
  decision:
  submission:
  sha256:

Recommendation:
  PROMOTE_EA_ENSEMBLE / KEEP_CURRENT_CONFORMER_SRFNET_ENSEMBLE / USE_EA_SRFNET_AS_SUPPLEMENT_ONLY / REJECT_EA

================================
```

---

## 18. 推荐执行顺序

不要一上来跑完整大实验。按下面顺序执行：

```text
Step 1. Final Freeze 当前 Conformer+SRFNet 主线，如果尚未冻结。
Step 2. 做 Source Inventory。
Step 3. 实现 EA 模块，并跑 unit test / diagnostics。
Step 4. 跑 fold_0 smoke：S0-S4。
Step 5. 只选择最多 2 个 EA 版本跑完整 5-fold。
Step 6. 做 Conformer + EA-SRFNet rank ensemble。
Step 7. 生成 public candidate，但不使用 public 反馈调参。
Step 8. 输出 final_candidate_board 和 final_recommendation。
```

---

## 19. 推荐默认命令模板

下面命令仅作为模板，实际路径和参数以项目脚本为准。

### 19.1 Baseline reproduce fold_0

```bash
python train_srfnet.py \
  --config path/to/trial0007_long_config.yaml \
  --folds 0 \
  --ea-mode none \
  --output-dir outputs/ea_srfnet_experiment/02_ea_srfnet_smoke/S0_baseline_reproduce
```

### 19.2 subject_self + raw_branch_only fold_0

```bash
python train_srfnet.py \
  --config path/to/trial0007_long_config.yaml \
  --folds 0 \
  --ea-mode subject_self \
  --ea-target raw_branch_only \
  --cache-ea \
  --output-dir outputs/ea_srfnet_experiment/02_ea_srfnet_smoke/S2_subject_self_raw_only
```

### 19.3 subject_self + all_input fold_0

```bash
python train_srfnet.py \
  --config path/to/trial0007_long_config.yaml \
  --folds 0 \
  --ea-mode subject_self \
  --ea-target all_input \
  --cache-ea \
  --output-dir outputs/ea_srfnet_experiment/02_ea_srfnet_smoke/S1_subject_self_all_input
```

### 19.4 EA_MAIN full 5-fold

```bash
python train_srfnet.py \
  --config path/to/trial0007_long_config.yaml \
  --folds all \
  --ea-mode subject_self \
  --ea-target raw_branch_only \
  --cache-ea \
  --output-dir outputs/ea_srfnet_experiment/03_ea_srfnet_5fold/EA_MAIN
```

---

## 20. 最终判断逻辑

EA 的最终价值不是“代码能跑”，而是下面这个判断：

```text
EA 是否能在不使用 public label、不改变 top-4 规则、不做 per-subject/per-fold oracle 的条件下，
让 SRFNet 或 Conformer+SRFNet 更稳、更强？
```

如果 EA 没有超过当前 `0.7958`：

```text
保持 Conformer + SRFNet rank ensemble 为主提交。
```

如果 EA 单模提升但 ensemble 没提升：

```text
EA-SRFNet 作为答辩补充和备选模型。
```

如果 EA ensemble 超过 `0.7958` 且审计通过：

```text
冻结 EA ensemble 为新主提交。
```

---

## 21. 给执行 AI 的最后提醒

```text
1. 当前主线已经很强，不要为了 EA 破坏现有结果。
2. EA 不是 AdaBN，不要混入 BN adaptation。
3. 优先 raw_branch_only，保护 FFT/log_power 分支。
4. 主线结论必须基于 original_trial + subject top-4；NoTop4 只能作为 Phase 8B 消融。
5. 所有主结论必须完整 5-fold。
6. public 只用于生成 submission，不参与模型选择。
7. 每一步都写 progress.md，便于人工接管。
```
