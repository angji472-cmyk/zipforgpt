# Step 1：冻结主线 + 工程正确性修复

> 交给 AI / Claude Code / Codex 的执行文件  
> 项目：BCIMI EEG Emotion Recognition  
> 阶段目标：先保住当前最优结果，再修掉会污染实验结论的工程问题。  
> 禁止：不要训练大模型，不要覆盖 PRIMARY，不要用 public/private feedback。

---

## 0. 当前背景

当前项目最强路线已经是：

```text
Conformer + SRFNet / S9_z_avg
subject-level Top-4 decoding
CV BA ≈ 0.8000
```

任务本质不是普通 trial-level 阈值分类，而是：

```text
每个 subject 有 8 个 trial
其中 4 个 positive，4 个 neutral
模型要在每个 subject 内把分数最高的 4 个 trial 选为 positive
```

本阶段只做：

```text
冻结当前 best
→ 审计提交格式
→ 修 250Hz/128Hz 不一致
→ 修 group-aware smoke 脚本 bug
→ 加 Top-4 / sampling rate / group 列名回归测试
→ 建 canonical leaderboard 雏形
```

---

## 1. 硬性规则

### 1.1 禁止事项

任何时候都禁止：

1. 覆盖当前 PRIMARY 提交文件
2. 删除旧 outputs
3. 使用 public/private test feedback 调参
4. 根据 fold_id 写规则
5. 根据 subject_id 写规则
6. 根据 trial_id / trial_index 写特判
7. 修改真实标签
8. 把 smoke 单折结果当成 full CV 结论
9. 训练大规模新模型

### 1.2 本阶段不做

本阶段不要做：

```text
Top-4 boundary loss 训练
Group-balanced sampler full 5-fold
多 seed ensemble
新 backbone
外部数据预训练
AdaBN / DANN / EA 大网格
```

这些放到 Step 2 和 Step 3。

---

## 2. Phase 0：冻结当前 PRIMARY

### 2.1 找到当前主线文件

请在仓库中定位当前 PRIMARY 相关文件，例如：

```text
outputs/final_submission_primary/
outputs/eeg_conformer_tuning/final_candidate/
outputs/srfnet/trial0007_long_e6e10/
outputs/*s9*z*avg*/
outputs/*candidate_board*
outputs/*final*audit*
```

建议搜索：

```bash
find outputs -iname "*s9*" -o -iname "*z_avg*" -o -iname "*primary*" -o -iname "*candidate*"
find outputs -iname "*.xlsx" -o -iname "*with_prob*.csv" -o -iname "*cv_predictions*.csv"
```

### 2.2 创建冻结目录

```bash
mkdir -p outputs/frozen_baselines/primary_$(date +%Y%m%d)
mkdir -p outputs/final_submission_primary
```

把当前 PRIMARY 的核心文件复制进去：

```text
frozen_primary_submission.xlsx
frozen_primary_with_prob.csv
frozen_primary_cv_predictions.csv
frozen_primary_candidate_board.md
frozen_primary_audit_report.md
```

如果文件名不一致，保留原始文件并额外复制一份标准命名。

### 2.3 生成 SHA256

```bash
cd outputs/frozen_baselines/primary_$(date +%Y%m%d)
sha256sum * > SHA256SUMS.txt
```

### 2.4 生成 manifest

创建：

```text
outputs/frozen_baselines/primary_YYYYMMDD/manifest.json
```

格式：

```json
{
  "name": "CURRENT_PRIMARY",
  "role": "PRIMARY_FROZEN_BASELINE",
  "ba": 0.8000,
  "decoding": "subject_level_top4",
  "models": ["Conformer", "SRFNet"],
  "ensemble": "z-score/rank/subject-level ensemble, use actual method from existing audit",
  "score_sources": [],
  "created_at": "",
  "files": {},
  "sha256": {},
  "notes": [
    "Do not overwrite this baseline.",
    "Any new candidate must beat this baseline under the same 5-fold protocol."
  ]
}
```

如果现有审计报告里写的是 BA=0.7958 而不是 0.8000，请以最新已通过审计的 PRIMARY 为准，并在 notes 里说明原因。

---

## 3. Phase 1：提交格式审计

### 3.1 审计目标

确认最终提交文件满足：

```text
80 rows
10 users
每 user 8 trials
每 user 恰好 4 个 Emotion_label=1
每 user 恰好 4 个 Emotion_label=0
无 user_id + trial_id 重复
Emotion_label 只能是 0 或 1
clean submission 不含多余列
```

### 3.2 新增脚本

创建：

```text
scripts/audit_submission_format.py
```

参考实现：

```python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["user_id", "trial_id", "Emotion_label"]


def audit_submission(path: str | Path) -> dict:
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    result = {
        "path": str(path),
        "n_rows": len(df),
        "columns": list(df.columns),
        "pass": True,
        "errors": [],
        "warnings": [],
    }

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            result["pass"] = False
            result["errors"].append(f"Missing required column: {col}")

    if not result["pass"]:
        return result

    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        result["warnings"].append(f"Extra columns found: {extra_cols}")

    if len(df) != 80:
        result["pass"] = False
        result["errors"].append(f"Expected 80 rows, got {len(df)}")

    dup = df.duplicated(["user_id", "trial_id"]).sum()
    if dup:
        result["pass"] = False
        result["errors"].append(f"Duplicated user_id+trial_id rows: {dup}")

    labels = set(df["Emotion_label"].dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        result["pass"] = False
        result["errors"].append(f"Invalid labels: {labels}")

    group = df.groupby("user_id")["Emotion_label"].agg(["count", "sum"])
    bad_count = group[group["count"] != 8]
    bad_pos = group[group["sum"] != 4]

    if len(group) != 10:
        result["pass"] = False
        result["errors"].append(f"Expected 10 users, got {len(group)}")

    if not bad_count.empty:
        result["pass"] = False
        result["errors"].append(f"Users without 8 trials: {bad_count.to_dict('index')}")

    if not bad_pos.empty:
        result["pass"] = False
        result["errors"].append(f"Users without exactly 4 positives: {bad_pos.to_dict('index')}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = audit_submission(args.path)

    lines = []
    lines.append("# Submission Format Audit")
    lines.append("")
    lines.append(f"- path: `{result['path']}`")
    lines.append(f"- pass: `{result['pass']}`")
    lines.append(f"- n_rows: `{result['n_rows']}`")
    lines.append(f"- columns: `{result['columns']}`")
    lines.append("")
    lines.append("## Errors")
    if result["errors"]:
        for e in result["errors"]:
            lines.append(f"- {e}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Warnings")
    if result["warnings"]:
        for w in result["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- None")

    text = "\n".join(lines)
    print(text)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
```

### 3.3 运行

```bash
python scripts/audit_submission_format.py   outputs/final_submission_primary/final_submission_primary.xlsx   --out outputs/final_submission_primary/submission_format_audit.md
```

如果实际文件名不同，替换路径。

---

## 4. Phase 2：修 250Hz / 128Hz 不一致

### 4.1 背景

本项目 EEG 采样率应为：

```text
SFREQ = 250Hz
4s window = 1000 samples
```

如果 bandpower / arena / feature branch 写死 128Hz，会导致频带划分错误，污染支线结论。

### 4.2 搜索

```bash
rg -n "128\.0|SFREQ = 128|band_fs: 128|fs: 128|sampling_rate.*128|sfreq.*128|n_times.*512" .
```

### 4.3 修改原则

优先从统一常量读取：

```python
from EEGNET.data.constants import SFREQ
```

或者在全局常量文件中定义：

```python
SFREQ = 250
WINDOW_SECONDS = 4
N_TIMES = SFREQ * WINDOW_SECONDS
```

不要在模型、脚本、yaml 里分散写死采样率。

### 4.4 重点检查文件

```text
EEGNET/data/constants.py
EEGNET/models/bandpower_layer.py
models/
scripts/arena_screening_v2.py
configs/*bandpower*.yaml
configs/*arena*.yaml
```

### 4.5 新增测试

创建：

```text
tests/test_sampling_rate_consistency.py
```

代码：

```python
def test_global_sampling_rate_is_250():
    from EEGNET.data.constants import SFREQ
    assert int(SFREQ) == 250
```

如果存在 BandPowerFeatureLayer：

```python
def test_bandpower_layer_uses_250hz():
    from EEGNET.data.constants import SFREQ
    from EEGNET.models.bandpower_layer import BandPowerFeatureLayer

    layer = BandPowerFeatureLayer(fs=SFREQ)
    assert int(layer.fs) == 250
```

### 4.6 输出报告

创建：

```text
reports/sfreq_consistency_audit.md
```

内容包含：

```text
1. 搜索命令
2. 搜到的 128Hz 位置
3. 已修改位置
4. 保留位置及原因
5. 测试结果
```

---

## 5. Phase 3：修 group-aware smoke 脚本

### 5.1 搜索

```bash
rg -n "gd\["pred"\]|sd\["pred"\]|\["pred"\]|y_pred|GroupBalancedSampler|group_embed|residual_head|gate" models scripts
```

### 5.2 修列名 pred / y_pred 不一致

如果预测函数输出列是：

```text
y_pred
```

则所有统计都必须用：

```python
balanced_accuracy_score(gd["y_true"], gd["y_pred"])
```

不要用：

```python
gd["pred"]
```

### 5.3 修 GroupBalancedSampler 先截断再随机的问题

错误逻辑：

```python
idxs = self.subj_indices[s][:windows_per_trial * 5]
if len(idxs) > windows_per_trial * 5:
    idxs = self.rng.choice(idxs, windows_per_trial * 5, replace=False).tolist()
```

正确逻辑：

```python
all_idxs = self.subj_indices[s]
limit = windows_per_trial * 5

if len(all_idxs) > limit:
    idxs = self.rng.choice(all_idxs, limit, replace=False).tolist()
else:
    idxs = list(all_idxs)
```

### 5.4 检查 group embedding 是否真的生效

如果存在：

```text
models/srfnet_group.py
```

检查是否出现：

```python
torch.cat([features, group_embedding], dim=-1)
...
features = features[:, :original_dim]
```

如果 group embedding 拼接后又被 slice 掉，则该 group 信息没有真正进入模型。

建议：

```text
不要优先修旧 SRFNetGroup
优先使用 SRFNetFiLM 作为 group-conditioned smoke
```

但本阶段只修 bug，不做训练。

### 5.5 输出报告

创建：

```text
reports/group_aware_bugfix_report.md
```

包含：

```text
1. 修复了哪些脚本
2. pred/y_pred 是否统一
3. sampler 是否真正随机抽样
4. group embedding 路径是否可信
5. 不启动 full training 的原因
```

---

## 6. Phase 4：Top-4 解码回归测试

### 6.1 新增测试

创建：

```text
tests/test_top4_decode.py
```

如果项目已有 `apply_subject_top4`，直接导入。否则写一个本地 helper 供测试。

代码示例：

```python
import pandas as pd


def simple_subject_top4(df, score_col="prob"):
    out = df.copy()
    out["y_pred"] = 0
    for sid, g in out.groupby("subject_id"):
        idx = g.sort_values(score_col, ascending=False).head(4).index
        out.loc[idx, "y_pred"] = 1
    return out


def test_subject_top4_selects_exactly_four():
    df = pd.DataFrame({
        "subject_id": ["S1"] * 8,
        "original_trial_id": list(range(1, 9)),
        "prob": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4],
        "y_true": [0, 1, 0, 1, 1, 0, 1, 0],
    })
    out = simple_subject_top4(df, score_col="prob")
    assert int(out["y_pred"].sum()) == 4


def test_subject_top4_multiple_subjects():
    df = pd.DataFrame({
        "subject_id": ["S1"] * 8 + ["S2"] * 8,
        "original_trial_id": list(range(1, 9)) * 2,
        "prob": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4] * 2,
        "y_true": [0, 1, 0, 1, 1, 0, 1, 0] * 2,
    })
    out = simple_subject_top4(df, score_col="prob")
    for sid, g in out.groupby("subject_id"):
        assert int(g["y_pred"].sum()) == 4
```

---

## 7. Phase 5：建立 canonical leaderboard

### 7.1 目标

当前项目里可能有多份 final report、candidate board、summary。需要建立一个统一 source of truth。

### 7.2 创建文件

```text
outputs/canonical_leaderboard.csv
outputs/canonical_leaderboard.md
```

字段：

```text
rank
candidate_name
role
mean_ba
dep_ba
hc_ba
fold0_ba
fold1_ba
fold2_ba
fold3_ba
fold4_ba
decoding
score_source
submission_file
audit_file
sha256
decision
notes
```

### 7.3 当前至少写入

```text
1. CURRENT_PRIMARY / S9_z_avg or Conformer+SRFNet
2. Conformer+exp3 rank ensemble
3. SRFNet long
4. tuned Conformer
5. exp3 frozen baseline
```

如果某些数值无法从本地文件复算，不要编造，写：

```text
MISSING_LOCAL_ARTIFACT
```

---

## 8. 最终输出

本阶段完成后，必须输出：

```text
outputs/frozen_baselines/primary_YYYYMMDD/manifest.json
outputs/frozen_baselines/primary_YYYYMMDD/SHA256SUMS.txt
outputs/final_submission_primary/submission_format_audit.md
reports/sfreq_consistency_audit.md
reports/group_aware_bugfix_report.md
outputs/canonical_leaderboard.csv
outputs/canonical_leaderboard.md
tests/test_sampling_rate_consistency.py
tests/test_top4_decode.py
```

---

## 9. 最终汇报模板

完成后生成：

```text
reports/step1_final_report.md
```

格式：

```markdown
# Step 1 Final Report

## Summary
- PRIMARY frozen: yes/no
- Submission audit: pass/fail
- SFREQ consistency: pass/fail
- Group-aware bugfix: pass/fail
- Top-4 tests: pass/fail
- Canonical leaderboard: created/not created

## Frozen Baseline
- Candidate:
- BA:
- DEP BA:
- HC BA:
- SHA256:

## Fixed Issues
| Issue | File | Status |
|---|---|---|

## Tests
| Test | Result |
|---|---|

## Blockers
- None / list blockers

## Decision
Step 1 complete. Ready for Step 2 Top-4 boundary and DEP robustness experiments.
```
