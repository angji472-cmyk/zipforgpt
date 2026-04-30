# Step 3：多 Seed Ensemble + 最终候选冻结与提交打包

> 交给 AI / Claude Code / Codex 的执行文件  
> 项目：BCIMI EEG Emotion Recognition  
> 阶段目标：基于 Step 1 的冻结主线和 Step 2 的候选结果，做多 seed Conformer/SRFNet ensemble、最终候选榜、提交文件审计和答辩材料。  
> 前置条件：必须完成 Step 1；Step 2 若无新候选，则直接围绕当前 PRIMARY 做多 seed / 打包。

---

## 0. 当前背景

当前项目最可信的提分路径是：

```text
subject-level Top-4
Conformer / SRFNet 排序信号
rank or z-score ensemble
多 seed 稳定化
```

本阶段不再大规模开发新模型，而是做：

```text
多 seed 训练或复用现有 seed
score 归一化
rank/z-score ensemble
candidate board
final submission audit
SHA256
答辩材料
```

---

## 1. 硬性规则

### 1.1 禁止事项

禁止：

1. 覆盖 Step 1 冻结 PRIMARY
2. 使用 public/private test feedback 调参
3. 用 fold_id / subject_id / trial_id 写特判
4. 改标签
5. 未审计就生成最终提交
6. 只凭单 seed 偶然涨分就替换主线
7. 只看 mean BA，不看 fold/group 稳定性

### 1.2 Promote 标准

最终新 PRIMARY 必须满足：

```text
mean BA > frozen primary
至少 3/5 folds 不低于 frozen primary
DEP BA 不下降
HC BA drop <= 0.01
Top-4 audit PASS
提交格式 PASS
score lineage 清楚
SHA256 已记录
```

---

## 2. Phase 0：收集候选输入

## 2.1 必须收集

从 Step 1：

```text
outputs/frozen_baselines/primary_YYYYMMDD/manifest.json
outputs/canonical_leaderboard.csv
```

从 Step 2：

```text
outputs/step2_experiments/step2_candidate_board.csv
outputs/step2_experiments/*/top4_predictions_cv.csv
outputs/step2_experiments/*/original_trial_scores.csv
```

从历史模型：

```text
Conformer original_trial_scores.csv
SRFNet original_trial_scores.csv
S9_z_avg / current primary scores
Conformer+exp3 backup scores
```

## 2.2 创建目录

```bash
mkdir -p outputs/step3_multiseed_ensemble
mkdir -p outputs/final_candidate_package
mkdir -p reports/final
```

---

## 3. Phase 1：多 seed 计划

## 3.1 优先级

如果算力有限，按这个顺序做：

```text
1. SRFNet 多 seed
2. Conformer 多 seed
3. Step 2 promoted candidate 多 seed
4. 旧 exp3 不优先
```

## 3.2 推荐 seeds

```text
2024
2025
2026
3407
42
```

如果时间紧，至少：

```text
2024
2025
3407
```

## 3.3 训练矩阵

### 情况 A：Step 2 没有新候选

训练：

```text
SRFNet baseline seed2024/2025/3407
Conformer tuned seed2024/2025/3407
```

### 情况 B：Step 2 有 promoted candidate

训练：

```text
Step2 best candidate seed2024/2025/3407
SRFNet baseline seed2024/2025/3407
Conformer tuned seed2024/2025/3407
```

## 3.4 输出要求

每个 seed 必须输出：

```text
original_trial_scores.csv
top4_predictions_cv.csv
fold_metrics.csv
group_metrics.csv
run_config.yaml
run_manifest.json
```

---

## 4. Phase 2：统一 score schema

所有模型的 original trial score 必须统一成以下列：

```text
model_name
seed
fold
subject_id
original_trial_id
y_true
score
prob
group
```

如果没有 `prob`，可以只用 `score`。  
如果没有 `group`，从 subject_id 推断：

```python
def infer_group(subject_id):
    s = str(subject_id).upper()
    if s.startswith("DEP"):
        return "DEP"
    if s.startswith("HC") or s.startswith("NOR") or s.startswith("NORMAL"):
        return "HC"
    return "UNKNOWN"
```

创建脚本：

```text
scripts/normalize_trial_scores.py
```

输出：

```text
outputs/step3_multiseed_ensemble/normalized_scores.csv
```

---

## 5. Phase 3：实现 ensemble 方法

创建脚本：

```text
scripts/run_step3_ensembles.py
```

至少实现以下方法：

## 5.1 mean_score

```text
直接平均原始 score
```

风险：不同模型 score scale 不一致。

## 5.2 mean_rank

每个 subject 内、每个模型把 8 个 trial 排名，再平均 rank。

```text
rank 越高越 positive
```

## 5.3 z_score_avg

对每个 subject、每个模型：

```text
z = (score - mean(score_subject_model)) / std(score_subject_model)
```

然后平均 z。

这是当前最值得优先复现的方法。

## 5.4 weighted_z_score_avg

搜索权重：

```text
Conformer weight: 0.30 到 0.70，步长 0.05
SRFNet weight: 1 - Conformer weight
```

如果有第三模型：

```text
只做小网格，不要大规模暴力搜索
```

## 5.5 robust_rank_aggregation

可选：

```text
median rank
trimmed mean rank
```

---

## 6. Phase 4：subject-level Top-4 解码

无论哪种 ensemble，最终都必须：

```text
每个 subject 内 score 最高 4 个 trial -> y_pred=1
其余 -> y_pred=0
```

实现：

```python
def apply_subject_top4(df, score_col="ensemble_score"):
    out = df.copy()
    out["y_pred"] = 0
    for sid, g in out.groupby("subject_id"):
        idx = g.sort_values(score_col, ascending=False).head(4).index
        out.loc[idx, "y_pred"] = 1
    return out
```

审计：

```text
每 subject 8 rows
每 subject sum(y_pred)=4
```

---

## 7. Phase 5：评估 ensemble

输出：

```text
outputs/step3_multiseed_ensemble/ensemble_board.csv
outputs/step3_multiseed_ensemble/ensemble_fold_metrics.csv
outputs/step3_multiseed_ensemble/ensemble_group_metrics.csv
outputs/step3_multiseed_ensemble/ensemble_changed_predictions.csv
outputs/step3_multiseed_ensemble/step3_ensemble_report.md
```

## 7.1 指标

必须包含：

```text
mean BA
Macro-F1
DEP BA
HC BA
fold0 BA
fold1 BA
fold2 BA
fold3 BA
fold4 BA
Top-4 audit
changed predictions vs frozen primary
```

## 7.2 changed predictions

对比 frozen primary：

```text
changed_count
changed_ratio
changed_by_subject
changed_by_fold
changed_by_group
```

如果 changed_ratio 极低但 BA 提升，说明候选较稳。  
如果 changed_ratio 很高但 BA 只小涨，风险较高。

---

## 8. Phase 6：最终候选决策

生成：

```text
outputs/final_candidate_package/final_candidate_board.md
outputs/final_candidate_package/final_candidate_board.csv
```

候选至少包括：

```text
PRIMARY
BACKUP_1
BACKUP_2
BACKUP_3
```

推荐结构：

```markdown
# Final Candidate Board

| Rank | Candidate | BA | DEP BA | HC BA | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | Role | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | PRIMARY | READY |
| 2 | ... | ... | ... | ... | ... | ... | ... | ... | ... | BACKUP_1 | READY |
```

## 8.1 决策规则

如果 Step 3 新 ensemble 满足 promote 标准：

```text
PROMOTE_NEW_PRIMARY
```

否则：

```text
KEEP_FROZEN_PRIMARY
```

不允许因为“看起来更复杂”就替换主线。

---

## 9. Phase 7：生成最终提交文件

## 9.1 公开/私有测试集推理

如果当前阶段要生成测试集提交，必须使用最终 PRIMARY 的同一套 pipeline：

```text
load model scores
normalize scores
ensemble
subject-level Top-4
write Excel
audit
SHA256
```

## 9.2 文件命名

```text
outputs/final_candidate_package/final_submission_PRIMARY.xlsx
outputs/final_candidate_package/final_submission_PRIMARY_with_score.csv
outputs/final_candidate_package/final_submission_BACKUP_1.xlsx
outputs/final_candidate_package/final_submission_BACKUP_1_with_score.csv
```

## 9.3 clean submission 格式

最终提交文件只保留：

```text
user_id
trial_id
Emotion_label
```

其中：

```text
Emotion_label = 1 positive
Emotion_label = 0 neutral
```

## 9.4 审计

运行 Step 1 的审计脚本：

```bash
python scripts/audit_submission_format.py   outputs/final_candidate_package/final_submission_PRIMARY.xlsx   --out outputs/final_candidate_package/final_submission_PRIMARY_audit.md
```

必须 PASS。

---

## 10. Phase 8：SHA256 与 manifest

生成：

```bash
cd outputs/final_candidate_package
sha256sum * > SHA256SUMS.txt
```

创建：

```text
outputs/final_candidate_package/final_manifest.json
```

格式：

```json
{
  "primary": {
    "name": "",
    "ba": 0.0,
    "dep_ba": 0.0,
    "hc_ba": 0.0,
    "fold_ba": [],
    "method": "",
    "decoding": "subject_level_top4",
    "submission_file": "",
    "with_score_file": "",
    "audit_file": "",
    "sha256": ""
  },
  "backups": [],
  "rules": {
    "uses_public_feedback": false,
    "uses_private_feedback": false,
    "uses_subject_id_rule": false,
    "uses_trial_index_rule": false,
    "changes_labels": false
  }
}
```

---

## 11. Phase 9：答辩材料

生成：

```text
reports/final/method_summary_1000words.md
reports/final/defense_talking_points.md
reports/final/ablation_table.md
reports/final/risk_and_caveats.md
```

## 11.1 method_summary_1000words.md

必须包含：

```text
1. 背景：跨被试 EEG 情绪识别
2. 挑战：subject 差异大，DEP/HC 差异明显，概率校准不稳定
3. 核心建模：subject-level Top-4 排序问题
4. 模型：Conformer + SRFNet
5. SRFNet：raw branch / FFT branch / regional feature / gate fusion
6. loss：BCE + ranking + consistency + optional Top-4 boundary
7. ensemble：rank/z-score ensemble
8. 结果：从 exp3 到 Conformer 到 SRFNet 到 ensemble 的提升链
9. 合规：未使用 public/private label，Top-4 来自赛题结构先验
```

## 11.2 defense_talking_points.md

至少回答：

### Q1：为什么可以用 Top-4？

答：

```text
赛题数据说明中每名测试 subject 有 8 段视频，其中 4 段中性、4 段积极。Top-4 解码使用的是任务结构先验，不是标签泄漏。
```

### Q2：为什么不是固定 threshold=0.5？

答：

```text
跨被试 EEG 的概率校准不稳定，但 subject 内相对排序更稳定。实验中 Top-4 明显优于 fixed threshold。
```

### Q3：为什么 ensemble？

答：

```text
Conformer 擅长时序局部-全局特征，SRFNet 融合 raw、频域和脑区统计特征，两者错误模式不完全相同。rank/z-score ensemble 能降低 score scale 差异。
```

### Q4：DEP 为什么难？

答：

```text
DEP 组 EEG 情绪反应可能与 HC 存在差异，且当前 group audit 显示 DEP BA 低于 HC。我们没有对 DEP fold0 写特判，而是用 group-balanced / robustness 方法进行受控探索。
```

### Q5：有没有过拟合 public？

答：

```text
没有使用 public/private label。所有模型选择基于训练集 5-fold CV。公开测试集若无标签，只可用于格式和分布审计，不用于参数选择。
```

---

## 12. Phase 10：最终报告

创建：

```text
reports/final/final_project_report.md
```

结构：

```markdown
# Final EEG Emotion Recognition Report

## 1. Final Decision
- PRIMARY:
- BACKUP_1:
- BACKUP_2:

## 2. Score Lineage
| Candidate | Score Source | Model | Seed | Fold | Notes |
|---|---|---|---|---|---|

## 3. Metrics
| Candidate | BA | DEP BA | HC BA | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## 4. Submission Audit
- rows:
- users:
- trials per user:
- 4/4 Top-4:
- duplicates:
- extra columns:
- SHA256:

## 5. Method Summary
- Top-4 decoding
- Conformer
- SRFNet
- z-score/rank ensemble

## 6. Ablations
| Method | BA | Decision |
|---|---:|---|

## 7. Caveats
- CV model selection bias
- Top-4 prior depends on competition data structure
- DEP remains harder than HC
- Small BA improvements need cautious interpretation

## 8. Final Recommendation
Submit PRIMARY. Preserve BACKUP_1 and BACKUP_2.
```

---

## 13. 最终输出清单

本阶段完成后，必须存在：

```text
outputs/step3_multiseed_ensemble/normalized_scores.csv
outputs/step3_multiseed_ensemble/ensemble_board.csv
outputs/step3_multiseed_ensemble/ensemble_fold_metrics.csv
outputs/step3_multiseed_ensemble/ensemble_group_metrics.csv
outputs/step3_multiseed_ensemble/ensemble_changed_predictions.csv
outputs/step3_multiseed_ensemble/step3_ensemble_report.md

outputs/final_candidate_package/final_candidate_board.csv
outputs/final_candidate_package/final_candidate_board.md
outputs/final_candidate_package/final_submission_PRIMARY.xlsx
outputs/final_candidate_package/final_submission_PRIMARY_with_score.csv
outputs/final_candidate_package/final_submission_PRIMARY_audit.md
outputs/final_candidate_package/SHA256SUMS.txt
outputs/final_candidate_package/final_manifest.json

reports/final/method_summary_1000words.md
reports/final/defense_talking_points.md
reports/final/ablation_table.md
reports/final/risk_and_caveats.md
reports/final/final_project_report.md
```

---

## 14. 结束判定

最终只允许以下两种结论：

### 情况 A：新 ensemble 成功

```text
PROMOTE_NEW_PRIMARY
原因：
- BA 超过 frozen primary
- >=3/5 folds 不下降
- DEP 不下降
- HC 下降 <=0.01
- Top-4 audit PASS
- submission audit PASS
```

### 情况 B：新 ensemble 没有稳定超过

```text
KEEP_FROZEN_PRIMARY
原因：
- 新方法没有稳定超过当前主线
- 当前主线已通过审计
- 新实验仅作为 supplementary / diagnostic
```

不要因为新方法更复杂就替换主线。
