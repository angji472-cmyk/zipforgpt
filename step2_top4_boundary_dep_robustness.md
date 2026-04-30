# Step 2：Top-4 Boundary Loss + DEP 鲁棒性实验

> 交给 AI / Claude Code / Codex 的执行文件  
> 项目：BCIMI EEG Emotion Recognition  
> 阶段目标：在不动当前 PRIMARY 的前提下，围绕“Top-4 排序目标”和“DEP group robustness”做受控 5-fold 实验。  
> 前置条件：必须先完成 Step 1 的冻结和工程正确性修复。

---

## 0. 当前背景

当前主线的关键认知：

```text
最终评估/提交不是 fixed threshold 二分类
而是 subject 内 8 个 trial 选 Top-4 positive
```

所以 SRFNet 训练目标需要更接近最终决策：

```text
当前：BCE + pairwise ranking + consistency
新增：Top-4 boundary loss
```

同时，当前项目的剩余短板主要来自 DEP 组：

```text
DEP BA < HC BA
DEP fold0 曾经明显失败
Conformer 和 SRFNet 在 DEP fold0 同时失败
简单 group-specific ensemble 没有突破
```

本阶段要做的是：

```text
实现 Top-4 boundary loss
实现 group-balanced sampler 实验
可选实现 FiLM smoke
做小规模、严格受控的 5-fold 实验矩阵
输出 fold/group/candidate board
```

---

## 1. 硬性规则

### 1.1 禁止事项

禁止：

1. 覆盖 Step 1 冻结的 PRIMARY
2. 使用 public/private feedback
3. 根据 fold_id / subject_id / trial_id 写规则
4. 修改标签
5. 只用 fold0 smoke 宣布成功
6. 只提升 DEP fold0 就 promote
7. HC 大幅下降但 overall 小涨也 promote
8. 没有 full 5-fold 就进入最终候选

### 1.2 Promote 标准

新候选必须满足：

```text
mean BA > 当前 PRIMARY
至少 3/5 folds 不低于当前 PRIMARY
DEP BA 不下降
HC BA 下降 <= 0.01
每 subject 仍满足 Top-4
无数据泄漏
```

如果只是在 mean BA 上打平，结论只能是：

```text
DIAGNOSTIC_ONLY_KEEP_MAINLINE
```

---

## 2. Phase 0：确认 Step 1 完成

运行：

```bash
pytest tests/test_sampling_rate_consistency.py
pytest tests/test_top4_decode.py
```

确认存在：

```text
outputs/frozen_baselines/primary_YYYYMMDD/manifest.json
outputs/canonical_leaderboard.csv
reports/step1_final_report.md
```

如果 Step 1 没完成，先停止本阶段。

---

## 3. Phase 1：实现 Top-4 Boundary Loss

## 3.1 新增文件

创建：

```text
training/losses/top4_boundary.py
```

代码：

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def top4_boundary_loss(
    trial_scores: torch.Tensor,
    trial_labels: torch.Tensor,
    subject_ids: list[str],
    *,
    tau: float = 0.2,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Subject-level Top-4 boundary surrogate.

    For each subject:
      hardest_pos = min(score of positive trials)
      hardest_neg = max(score of neutral trials)

    The loss encourages:
      hardest_pos > hardest_neg

    This directly targets the rank4/rank5 boundary that determines Top-4 decoding.
    """
    if trial_scores.ndim != 1:
        trial_scores = trial_scores.view(-1)
    if trial_labels.ndim != 1:
        trial_labels = trial_labels.view(-1)

    losses = []
    seen = list(dict.fromkeys(subject_ids))

    for sid in seen:
        idx = [i for i, s in enumerate(subject_ids) if s == sid]
        if not idx:
            continue

        idx_t = torch.as_tensor(idx, device=trial_scores.device, dtype=torch.long)
        s = trial_scores.index_select(0, idx_t)
        y = trial_labels.index_select(0, idx_t) > 0.5

        pos = s[y]
        neg = s[~y]

        if pos.numel() == 0 or neg.numel() == 0:
            continue

        hardest_pos = pos.min()
        hardest_neg = neg.max()
        losses.append(F.softplus((hardest_neg - hardest_pos + margin) / tau))

    if not losses:
        return trial_scores.new_tensor(0.0)

    return torch.stack(losses).mean()
```

---

## 3.2 新增单元测试

创建：

```text
tests/test_top4_boundary_loss.py
```

代码：

```python
import torch

from training.losses.top4_boundary import top4_boundary_loss


def test_top4_boundary_loss_lower_when_ranking_correct():
    subject_ids = ["S1"] * 8
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)

    good_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    bad_scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.4, 0.1])

    good_loss = top4_boundary_loss(good_scores, labels, subject_ids)
    bad_loss = top4_boundary_loss(bad_scores, labels, subject_ids)

    assert good_loss < bad_loss


def test_top4_boundary_loss_handles_missing_pairs():
    subject_ids = ["S1"] * 4
    labels = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])

    loss = top4_boundary_loss(scores, labels, subject_ids)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
```

运行：

```bash
pytest tests/test_top4_boundary_loss.py
```

---

## 4. Phase 2：接入 SRFNet 训练

## 4.1 找到训练脚本

优先检查：

```text
scripts/train_srfnet.py
train_srfnet.py
```

搜索：

```bash
rg -n "BCEWithLogits|pairwise|consistency|loss =" scripts train_srfnet.py training
```

## 4.2 配置项

在 SRFNet config 中加入：

```yaml
loss:
  bce_weight: 1.0
  rank_weight: 0.1
  consistency_weight: 0.05
  top4_boundary_weight: 0.0
  top4_boundary_tau: 0.2
  top4_boundary_margin: 0.0
```

如果原配置字段名不同，沿用原风格，但必须包含：

```text
top4_boundary_weight
top4_boundary_tau
top4_boundary_margin
```

## 4.3 训练总 loss

接入方式：

```python
from training.losses.top4_boundary import top4_boundary_loss

loss_top4 = top4_boundary_loss(
    trial_scores=trial_scores,
    trial_labels=trial_labels,
    subject_ids=subject_ids,
    tau=cfg.loss.top4_boundary_tau,
    margin=cfg.loss.top4_boundary_margin,
)

loss = (
    bce_weight * loss_bce
    + rank_weight * loss_rank
    + consistency_weight * loss_consistency
    + top4_boundary_weight * loss_top4
)
```

### 重要注意

如果当前 batch 不是完整 subject 的 8 个 original trials，则不要在 window-level mini-batch 上错误计算 Top-4 boundary。

优先方案：

```text
在 original_trial 聚合后的 subject batch 上计算 top4_boundary_loss
```

如果训练循环无法轻松保证完整 subject，则先做：

```text
subject-complete sampler
或 epoch-end auxiliary top4 loss
```

不允许在不完整 subject 数据上伪造 Top-4 边界。

---

## 5. Phase 3：配置实验矩阵

创建配置目录：

```text
configs/experiments_step2/
```

至少创建以下配置：

```text
srfnet_baseline_reproduce.yaml
srfnet_top4_w005.yaml
srfnet_top4_w010.yaml
srfnet_top4_w020.yaml
srfnet_group_balanced.yaml
srfnet_top4_w010_group_balanced.yaml
```

## 5.1 baseline 复现

```yaml
name: srfnet_baseline_reproduce
loss:
  top4_boundary_weight: 0.0
sampler:
  group_balanced: false
```

## 5.2 Top-4 boundary ablation

```yaml
name: srfnet_top4_w005
loss:
  top4_boundary_weight: 0.05
  top4_boundary_tau: 0.2
  top4_boundary_margin: 0.0
```

```yaml
name: srfnet_top4_w010
loss:
  top4_boundary_weight: 0.10
  top4_boundary_tau: 0.2
  top4_boundary_margin: 0.0
```

```yaml
name: srfnet_top4_w020
loss:
  top4_boundary_weight: 0.20
  top4_boundary_tau: 0.2
  top4_boundary_margin: 0.0
```

## 5.3 Group-balanced sampler

```yaml
name: srfnet_group_balanced
sampler:
  group_balanced: true
  groups:
    - DEP
    - HC
```

## 5.4 Combined

```yaml
name: srfnet_top4_w010_group_balanced
loss:
  top4_boundary_weight: 0.10
  top4_boundary_tau: 0.2
  top4_boundary_margin: 0.0
sampler:
  group_balanced: true
```

---

## 6. Phase 4：实现 Group-Balanced Sampler

## 6.1 目标

让每个 batch 尽量平衡 DEP / HC subject，避免模型过度偏向 HC。

数据结构：

```text
DEP: 20 subjects
HC: 40 subjects
每 fold:
  DEP 4 subjects
  HC 8 subjects
```

## 6.2 实现原则

优先按 subject 采样，而不是按 window 盲采样。

理想 batch 结构：

```text
每 batch:
  若干 DEP subjects
  若干 HC subjects
  每个 subject 采样相同数量 windows/trials
```

## 6.3 注意

不要在推理时输入 group label。

本实验目标是训练采样平衡，而不是部署依赖 group metadata。

---

## 7. Phase 5：运行实验

## 7.1 先做 fold0 smoke

每个配置先跑 fold0：

```bash
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w005.yaml --fold 0
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w010.yaml --fold 0
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w020.yaml --fold 0
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_group_balanced.yaml --fold 0
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w010_group_balanced.yaml --fold 0
```

## 7.2 Smoke 通过标准

fold0 smoke 只用于排除明显失败。

通过条件：

```text
不崩溃
Top-4 decode 正常
fold0 BA 没有明显大跌
loss 数值稳定
无 NaN
```

不要因为 fold0 提升就 promote。

## 7.3 Full 5-fold

对 smoke 没崩的配置跑 full 5-fold：

```bash
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w005.yaml --folds 0,1,2,3,4
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w010.yaml --folds 0,1,2,3,4
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w020.yaml --folds 0,1,2,3,4
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_group_balanced.yaml --folds 0,1,2,3,4
python scripts/train_srfnet.py --config configs/experiments_step2/srfnet_top4_w010_group_balanced.yaml --folds 0,1,2,3,4
```

如果训练脚本参数名不同，请适配实际仓库。

---

## 8. Phase 6：统一评估脚本

创建：

```text
scripts/evaluate_step2_candidates.py
```

输入：

```text
outputs/step2_experiments/*/top4_predictions_cv.csv
```

输出：

```text
outputs/step2_experiments/step2_candidate_board.csv
outputs/step2_experiments/step2_group_metrics.csv
outputs/step2_experiments/step2_fold_metrics.csv
outputs/step2_experiments/step2_final_report.md
```

指标：

```text
mean BA
Macro-F1
AUC if available
DEP BA
HC BA
fold0 BA
fold1 BA
fold2 BA
fold3 BA
fold4 BA
subject-level Top-4 audit
changed predictions vs primary if available
```

---

## 9. Phase 7：决策逻辑

对于每个候选，给出以下决策之一：

```text
PROMOTE_TO_STEP3_CANDIDATE
DIAGNOSTIC_ONLY_KEEP_MAINLINE
REJECT_REGRESSION
NEEDS_RERUN
```

### 9.1 PROMOTE 条件

```text
mean BA > current primary
>= 3/5 folds 不下降
DEP BA 不下降
HC BA drop <= 0.01
Top-4 audit PASS
```

### 9.2 DIAGNOSTIC_ONLY

满足任一情况：

```text
打平 primary，但没超过
只提升 DEP，overall 没提升
只提升 fold0
DEP 提升但 HC 下降明显
```

### 9.3 REJECT

满足任一情况：

```text
mean BA 明显下降
Top-4 audit failed
训练不稳定
NaN
fold 表现极不稳定
```

---

## 10. 可选 Phase：FiLM smoke

只有在 group-balanced sampler 修复后，才做 FiLM smoke。

### 10.1 目标

用极小参数量建模 DEP/HC 差异：

```text
h' = gamma_group * h + beta_group
```

### 10.2 原则

```text
near-identity initialization
只作用于高层 feature
不要作用于 raw EEG 输入
不要在推理时依赖未知 test group，除非测试文件明确提供 group
```

### 10.3 输出

```text
outputs/step2_film_smoke/
reports/film_smoke_report.md
```

如果 fold0 smoke 低于 baseline 明显，不进入 5-fold。

---

## 11. 最终输出

本阶段必须输出：

```text
training/losses/top4_boundary.py
tests/test_top4_boundary_loss.py
configs/experiments_step2/srfnet_baseline_reproduce.yaml
configs/experiments_step2/srfnet_top4_w005.yaml
configs/experiments_step2/srfnet_top4_w010.yaml
configs/experiments_step2/srfnet_top4_w020.yaml
configs/experiments_step2/srfnet_group_balanced.yaml
configs/experiments_step2/srfnet_top4_w010_group_balanced.yaml
outputs/step2_experiments/step2_candidate_board.csv
outputs/step2_experiments/step2_group_metrics.csv
outputs/step2_experiments/step2_fold_metrics.csv
outputs/step2_experiments/step2_final_report.md
```

---

## 12. 最终报告模板

创建：

```text
reports/step2_final_report.md
```

内容：

```markdown
# Step 2 Final Report

## Summary
- Best new candidate:
- Current primary BA:
- Best candidate BA:
- Decision:

## Experiment Board
| Candidate | BA | DEP BA | HC BA | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Top-4 Boundary Loss Findings
- w=0.05:
- w=0.10:
- w=0.20:

## Group-Balanced Findings
- DEP effect:
- HC effect:
- Stability:

## Top-4 Audit
- All candidates pass/fail:
- Bad subjects if any:

## Promote Decision
- PROMOTE_TO_STEP3_CANDIDATE / DIAGNOSTIC_ONLY_KEEP_MAINLINE / REJECT

## Next Step
Proceed to Step 3 only with promoted candidates.
```
