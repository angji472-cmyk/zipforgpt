# ST-DADGAT-lite 实验任务书：DE/PSD/STFT8 + Channel Graph Attention + Temporal Attention + MMD Domain Loss

## 0. 任务目标

请在当前 EEG 情绪识别项目中实现并评估一个 **ST-DADGAT-lite** 模型，用于验证以下路线是否能超过当前最强可信 baseline：

```text
DE / PSD / STFT8 features
    ↓
channel graph attention
    ↓
temporal attention
    ↓
classifier
    ↓
MMD domain loss for subject-invariant representation
```

当前主 baseline：

```text
exp3_vote_5fold
BA = 0.6329 ± 0.0191
Macro-F1 = 0.6285 ± 0.0232
ROC-AUC = 0.6883 ± 0.0211
```

本任务不是做完整 ST-DADGAT 论文复现，而是做一个 **工程上可控、低泄露风险、可公平比较的 lite 版本**。

最终目标：

1. 使用与 `exp3_vote_5fold` 完全相同的 5-fold cross-subject split。
2. 使用 DE / PSD / STFT8 时频特征。
3. 实现轻量 channel graph attention。
4. 实现轻量 temporal attention。
5. 加入 MMD domain loss，使 representation 更 subject-invariant。
6. 与 `exp3_vote_5fold` 进行公平对比。
7. 判断该路线是否值得继续深入。

---

## 1. 严格约束

### 1.1 禁止事项

1. 不要修改已有实验结果。
2. 不要覆盖 `exp3_vote_5fold`。
3. 不要使用 public test 数据训练、调参或做 MMD。
4. 不要把 validation/test subject 的有标签信息用于训练。
5. 不要在 split 前 fit scaler / normalizer / PCA / feature selector。
6. 不要先对全数据做 oversampling / augmentation / feature selection 后再划分。
7. 不要用 test/public 指标选 checkpoint。
8. 不要把 transductive 结果混入 inductive leaderboard。
9. 不要删除任何旧文件。

### 1.2 必须遵守

1. 所有新代码写在新文件中，不破坏原 pipeline。
2. 所有新结果放到新目录：

```text
outputs/st_dadgat_lite/
```

3. 所有 split 必须复用 `exp3_vote_5fold` 的 subject-level split。
4. 每个 fold 内独立 fit scaler / normalizer。
5. MMD 只能在 training fold 内使用。
6. 每个 fold 都要保存：
   - config
   - metrics
   - predictions
   - split hash
   - training log
   - model checkpoint
7. 最终必须输出完整 5-fold mean ± std。
8. 最终必须生成对比报告。

---

## 2. 推荐目录结构

请新增以下文件或目录。实际路径可根据当前项目结构调整，但必须保持清晰。

```text
ST_DADGAT_LITE/
├── configs/
│   └── st_dadgat_lite.yaml
├── data/
│   └── stf_feature_dataset.py
├── models/
│   ├── st_dadgat_lite.py
│   ├── graph_attention.py
│   ├── temporal_attention.py
│   └── mmd_loss.py
├── training/
│   └── train_st_dadgat_lite.py
├── evaluation/
│   └── evaluate_st_dadgat_lite.py
├── scripts/
│   ├── run_st_dadgat_lite_5fold.py
│   ├── audit_st_dadgat_lite.py
│   └── compare_with_exp3_vote.py
└── outputs/
    └── st_dadgat_lite/
```

如果现有工程已有类似目录，例如 `EEGNET/` 或 `EEGNET+AdaBN+DANN/`，也可以直接放在现有工程下，例如：

```text
EEGNET/st_dadgat_lite/
```

但不得覆盖旧代码。

---

## 3. 实验命名

主实验名：

```text
st_dadgat_lite_mmd_5fold
```

消融实验名：

```text
stf_logreg_baseline_5fold
stf_mlp_no_graph_no_mmd_5fold
st_dadgat_lite_no_mmd_5fold
st_dadgat_lite_mmd_5fold
```

输出目录建议：

```text
outputs/st_dadgat_lite/stf_logreg_baseline_5fold/
outputs/st_dadgat_lite/stf_mlp_no_graph_no_mmd_5fold/
outputs/st_dadgat_lite/st_dadgat_lite_no_mmd_5fold/
outputs/st_dadgat_lite/st_dadgat_lite_mmd_5fold/
```

---

## 4. 数据与特征

### 4.1 输入特征

使用以下特征：

```text
DE
PSD
STFT8
```

其中：

- DE = differential entropy features
- PSD = power spectral density / band power features
- STFT8 = 8 个 time bins 的时频特征

不要重新发明复杂特征。优先复用当前项目中已有的 DE / PSD / STFT8 特征生成代码。

### 4.2 特征形状建议

建议将每个样本表示成：

```text
X.shape = [channels, time_bins, feature_dim]
```

例如：

```text
channels = EEG channel count
time_bins = 8
feature_dim = band/features per channel per time bin
```

如果 DE / PSD 是静态特征，可复制到每个 time bin，或者作为 channel-level static embedding 与 STFT8 拼接。

推荐统一成：

```text
X = concat([
    DE_repeated_over_time,
    PSD_repeated_over_time,
    STFT8
], dim=-1)
```

最终输入：

```text
[B, C, T, F]
```

其中：

```text
B = batch size
C = channels
T = 8
F = per-channel feature dim
```

### 4.3 标准化要求

每个 fold 内：

```python
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
```

禁止：

```python
scaler.fit(all_data)
```

如果使用 per-channel normalization，也必须只用 training fold 统计量。

---

## 5. Split 要求

### 5.1 复用 exp3_vote_5fold

必须复用当前最强 baseline 的 5-fold subject split：

```text
exp3_vote_5fold
```

每个 fold 必须记录：

```text
fold_id
train_subjects
val_subjects / test_subjects
train_sample_ids
val_sample_ids
split_hash
```

### 5.2 split_hash

生成 split_hash：

```python
split_hash = hash(sorted(train_ids) + sorted(val_ids) + sorted(test_ids))
```

如果当前 pipeline 没有 `test_ids`，只用 train/val 也可以，但要在报告中说明。

### 5.3 泄露检查

每个 fold 训练前必须检查：

1. train subjects 与 val/test subjects 无重叠；
2. train trial_id 与 val/test trial_id 无重叠；
3. train window_id 与 val/test window_id 无重叠；
4. public samples 未进入训练；
5. scaler 只 fit training data。

若发现任何重叠，立即停止该 fold，并写入审计报告。

---

## 6. 模型结构

## 6.1 总体结构

```text
Input X: [B, C, T, F]
    ↓
feature projection
    ↓
channel graph attention per time bin
    ↓
temporal attention across T
    ↓
subject-invariant representation z
    ↓
classifier
```

---

## 6.2 Feature Projection

将每个 channel-time feature 投影到 hidden dim：

```python
h = Linear(F, d_model)(X)
```

输入输出：

```text
X: [B, C, T, F]
h: [B, C, T, d_model]
```

推荐参数：

```yaml
d_model: 64
dropout: 0.3
activation: gelu
```

---

## 6.3 Channel Graph Attention

### 6.3.1 目标

对每个 time bin，在 channel 维度上做图注意力：

```text
每个 EEG channel 是一个 node
channel 间的边可以是 learnable adjacency 或 fixed adjacency
```

### 6.3.2 Lite 实现优先级

优先实现 learnable dense adjacency，不要一开始就引入复杂脑区先验。

对于每个 time bin：

```text
H_t: [B, C, d_model]
GraphAttention(H_t) -> [B, C, d_model]
```

可以用以下任一实现：

#### 实现 A：简化版 GAT

```python
Q = Wq(H)
K = Wk(H)
V = Wv(H)

attn = softmax((Q @ K.transpose(-1, -2)) / sqrt(d))
out = attn @ V
```

这是 channel self-attention，本质上是 learnable graph attention。

#### 实现 B：加入可学习 adjacency bias

```python
attn_logits = Q @ K.T / sqrt(d) + A_bias
attn = softmax(attn_logits)
out = attn @ V
```

其中：

```python
A_bias: [C, C]
```

初始化为 0。

推荐先做实现 B。

### 6.3.3 输出

对所有 time bins 处理后：

```text
H_graph: [B, C, T, d_model]
```

---

## 6.4 Temporal Attention

### 6.4.1 目标

建模 8 个 STFT time bins 之间的动态关系。

### 6.4.2 推荐实现

先对 channel 维度 pooling：

```python
H_time = mean(H_graph, dim=channel)
```

得到：

```text
H_time: [B, T, d_model]
```

然后使用 TransformerEncoder 或 MultiHeadAttention：

```python
TemporalSelfAttention(H_time)
```

推荐参数：

```yaml
temporal_heads: 4
temporal_layers: 1
dropout: 0.3
```

输出：

```text
H_temporal: [B, T, d_model]
```

最后 pooling：

```python
z = mean(H_temporal, dim=time)
```

得到：

```text
z: [B, d_model]
```

---

## 6.5 Classifier

```python
logits = MLP(z)
```

推荐：

```text
Linear(d_model, 64)
GELU
Dropout(0.3)
Linear(64, num_classes)
```

---

## 6.6 Domain Head 可选

本任务主推 MMD，不强制 DANN。

不要做 gradient reversal，除非 MMD 完成且稳定。

---

## 7. MMD Domain Loss

## 7.1 目标

让不同 subject 的 representation 分布更接近，从而提升 cross-subject 泛化。

MMD 作用在：

```text
z: [B, d_model]
```

### 7.2 严格限制

MMD 只能在 training fold 内计算。

禁止使用 validation/test/public subject 的 representation 参与 MMD。

### 7.3 推荐 subject pairing 方式

每个 batch 内包含多个 subject。

计算：

```text
classification loss
+
lambda_mmd * average pairwise MMD across subjects in batch
```

如果 batch 中只有一个 subject，则 MMD loss = 0。

### 7.4 MMD 实现

使用 RBF kernel MMD：

```python
def rbf_kernel(x, y, sigma):
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_rbf(x, y, sigmas=(1, 2, 4, 8, 16)):
    loss = 0
    for sigma in sigmas:
        k_xx = rbf_kernel(x, x, sigma).mean()
        k_yy = rbf_kernel(y, y, sigma).mean()
        k_xy = rbf_kernel(x, y, sigma).mean()
        loss += k_xx + k_yy - 2 * k_xy
    return loss / len(sigmas)
```

### 7.5 Batch 内多 subject MMD

```python
def subject_mmd_loss(z, subject_ids):
    unique_subjects = subject_ids.unique()
    if len(unique_subjects) < 2:
        return z.new_tensor(0.0)

    losses = []
    for i in range(len(unique_subjects)):
        for j in range(i + 1, len(unique_subjects)):
            zi = z[subject_ids == unique_subjects[i]]
            zj = z[subject_ids == unique_subjects[j]]
            if len(zi) >= 2 and len(zj) >= 2:
                losses.append(mmd_rbf(zi, zj))

    if not losses:
        return z.new_tensor(0.0)

    return torch.stack(losses).mean()
```

### 7.6 Loss

```python
loss = ce_loss + lambda_mmd * mmd_loss
```

推荐先用：

```yaml
lambda_mmd: 0.01
```

然后做小网格：

```yaml
lambda_mmd_grid: [0.0, 0.001, 0.005, 0.01, 0.02]
```

注意：

- `lambda_mmd=0.0` 就是 no-MMD 消融；
- 最终不能只报调参最优，要说明这是模型选择结果；
- 如果使用同一 CV 调 `lambda_mmd`，结论只能作为阶段性模型选择，不是严格无偏泛化估计。

---

## 8. 训练配置

推荐配置：

```yaml
experiment:
  name: st_dadgat_lite_mmd_5fold
  seed: 42
  folds: 5
  split_source: exp3_vote_5fold

data:
  features:
    - DE
    - PSD
    - STFT8
  metric_scope: trial10s/all:all
  normalize: true
  scaler_fit_scope: train_fold_only

model:
  name: ST_DADGAT_Lite
  d_model: 64
  graph_heads: 4
  temporal_heads: 4
  temporal_layers: 1
  dropout: 0.3
  classifier_hidden: 64
  adjacency_bias: true

training:
  batch_size: 32
  epochs: 80
  optimizer: AdamW
  lr: 0.0005
  weight_decay: 0.01
  scheduler: cosine
  early_stopping: true
  patience: 15
  monitor: val_balanced_accuracy
  save_best: true
  gradient_clip_norm: 1.0

loss:
  ce_weight: 1.0
  mmd_weight: 0.01
  mmd_sigmas: [1, 2, 4, 8, 16]

evaluation:
  metrics:
    - balanced_accuracy
    - macro_f1
    - roc_auc
  aggregate:
    - mean
    - std
```

---

## 9. 必须做的实验

## 9.1 E0：复用当前最佳 baseline

不需要重新训练，但必须读取并记录：

```text
exp3_vote_5fold
BA = 0.6329 ± 0.0191
Macro-F1 = 0.6285 ± 0.0232
ROC-AUC = 0.6883 ± 0.0211
```

作为 comparison baseline。

---

## 9.2 E1：STF LogReg baseline

复现或读取已有结果：

```text
DE+PSD+STFT8 + LogReg
```

已有参考：

```text
BA = 0.6238 ± 0.0058
```

用途：

- 检查特征质量；
- 作为 hand-crafted feature baseline。

---

## 9.3 E2：STF MLP baseline

实现：

```text
DE+PSD+STFT8
+ MLP
无 graph
无 temporal attention
无 MMD
```

目的：

判断简单神经网络是否已经能充分利用 STF 特征。

---

## 9.4 E3：Graph + Temporal，无 MMD

实现：

```text
DE+PSD+STFT8
+ channel graph attention
+ temporal attention
+ classifier
无 MMD
```

目的：

判断图注意力 + 时间注意力是否本身有效。

---

## 9.5 E4：Graph + Temporal + MMD

实现：

```text
DE+PSD+STFT8
+ channel graph attention
+ temporal attention
+ MMD domain loss
```

目的：

判断 subject-invariant MMD 是否能提升 cross-subject 泛化。

---

## 9.6 E5：MMD 权重小网格

只在 E4 基础上做小网格：

```text
lambda_mmd ∈ [0.0, 0.001, 0.005, 0.01, 0.02]
```

必须记录每个 lambda 的 5-fold 结果。

---

## 10. 通过标准

一个实验要成为主线候选，必须满足：

```text
BA >= 0.6379
```

或：

```text
Macro-F1 >= 0.6335
```

并且同时满足：

1. 5-fold CV；
2. 无 subject/trial/window 泄露；
3. public 不参与训练或指标计算；
4. std 不明显高于 baseline；
5. 至少 3/5 folds 相比 `exp3_vote_5fold` 有提升；
6. 与 baseline 使用相同 split；
7. 没有 test/public 调参；
8. MMD 只使用 training fold 内 subject。

如果没有达到上述条件，则不能替代 `exp3_vote_5fold`。

---

## 11. 输出文件

每个实验目录必须包含：

```text
config.yaml
summary_metrics.json
fold_metrics.csv
predictions.csv
classification_report.csv
confusion_matrix.csv
training_log.csv
split_integrity.json
leakage_check.json
```

最终总输出目录：

```text
outputs/st_dadgat_lite/final_report/
```

必须生成：

```text
st_dadgat_lite_report.md
st_dadgat_lite_leaderboard.csv
st_dadgat_lite_ablation.csv
st_dadgat_lite_vs_exp3_vote.csv
st_dadgat_lite_leakage_audit.csv
mmd_lambda_sweep.csv
```

---

## 12. 报告模板

最终 `st_dadgat_lite_report.md` 至少包含以下内容：

```markdown
# ST-DADGAT-lite 实验报告

## 1. 实验目标

## 2. 数据与 split
- 是否复用 exp3_vote_5fold split
- 每个 fold 的 subject 数
- split hash
- 泄露检查结果

## 3. 模型结构
- DE/PSD/STFT8 输入
- channel graph attention
- temporal attention
- MMD loss

## 4. 实验结果

| Experiment | BA | Macro-F1 | ROC-AUC | ΔBA vs exp3 | ΔMF1 vs exp3 |
|---|---:|---:|---:|---:|---:|

## 5. Per-fold 对比

| Fold | exp3_vote BA | ST-DADGAT-lite BA | Delta |
|---|---:|---:|---:|

## 6. MMD ablation

| lambda_mmd | BA | Macro-F1 | ROC-AUC | Notes |
|---:|---:|---:|---:|---|

## 7. 泄露审计
- subject overlap
- trial/window overlap
- scaler fit scope
- MMD training scope
- public usage

## 8. 结论
- 是否超过 exp3_vote_5fold
- 是否进入主线候选
- 是否建议继续 Graph/Transformer/DA 路线
```

---

## 13. 最终终端摘要

任务结束时，在终端打印：

```text
===== ST-DADGAT-lite Summary =====

Baseline exp3_vote_5fold:
  BA:
  Macro-F1:
  ROC-AUC:

Best ST-DADGAT-lite:
  Experiment:
  BA:
  Macro-F1:
  ROC-AUC:
  Delta BA:
  Delta Macro-F1:
  Improved folds:

Leakage:
  subject overlap:
  trial overlap:
  window overlap:
  scaler leakage:
  public used:
  MMD uses val/test/public:

Mainline candidate:
  YES/NO

Recommendation:
  KEEP exp3_vote_5fold / PROMOTE ST_DADGAT_Lite / NEED MORE TESTING

Report:
  outputs/st_dadgat_lite/final_report/st_dadgat_lite_report.md
=================================
```

---

## 14. 风险解释

请在报告中明确说明：

1. 如果 ST-DADGAT-lite 没超过 `exp3_vote_5fold`，这并不说明 graph/transformer/domain adaptation 方向无效，只说明当前 lite 实现和当前数据设置下没有超过。
2. 如果 MMD 有提升但仍未超过 baseline，则可以作为后续增强方向。
3. 如果 MMD 降低性能，说明当前 subject distribution alignment 可能破坏类别判别边界。
4. 如果 Graph + Temporal 提升明显但 MMD 不提升，后续可以保留 encoder、去掉 MMD。
5. 如果 LogReg 仍然接近最佳，说明特征空间可能已经饱和，复杂模型容易过拟合。
6. 如果 std 明显增大，即使 mean 提升，也不能直接推广为更强模型。

---

## 15. 最终决策规则

### 情况 A：显著超过 baseline

条件：

```text
BA >= 0.6379
Macro-F1 >= 0.6335
>= 3/5 folds improved
std 不明显增大
无泄露
```

结论：

```text
ST-DADGAT-lite 可进入主线候选。
```

---

### 情况 B：接近但未超过

条件：

```text
0.625 <= BA < 0.6379
```

结论：

```text
ST-DADGAT-lite 可作为补充候选，但不替代 exp3_vote_5fold。
```

---

### 情况 C：明显低于 baseline

条件：

```text
BA < 0.625
```

结论：

```text
ST-DADGAT-lite 当前实现不推荐继续扩大调参，应回到 exp3_vote_5fold 或尝试 backbone-level contrastive learning。
```

---

## 16. 优先级提醒

请先跑最小闭环，不要一开始堆复杂模块。

推荐执行顺序：

```text
Step 1: 确认 split 复用 exp3_vote_5fold
Step 2: 构造 DE/PSD/STFT8 Dataset
Step 3: 跑 STF MLP baseline
Step 4: 加 Channel Graph Attention
Step 5: 加 Temporal Attention
Step 6: 加 MMD
Step 7: 做 lambda_mmd 小网格
Step 8: 生成报告
```

如果 Step 3 的 STF MLP 远低于 LogReg 或 exp3，则先不要继续堆复杂模型，先检查特征、归一化、split 和 label 对齐。

---

## 17. 交付标准

任务完成后，请确保：

```text
outputs/st_dadgat_lite/final_report/st_dadgat_lite_report.md
outputs/st_dadgat_lite/final_report/st_dadgat_lite_leaderboard.csv
outputs/st_dadgat_lite/final_report/st_dadgat_lite_ablation.csv
outputs/st_dadgat_lite/final_report/st_dadgat_lite_vs_exp3_vote.csv
outputs/st_dadgat_lite/final_report/st_dadgat_lite_leakage_audit.csv
outputs/st_dadgat_lite/final_report/mmd_lambda_sweep.csv
```

全部存在。

如果某项未完成，请在报告中明确说明：

```text
未完成项
原因
是否影响最终结论
下一步建议
```
