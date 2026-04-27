# EEG 情绪识别外部预训练任务书：DE/PSD/STFT8 + Graph/Transformer Pretrain → Fine-tune

## 0. 任务背景

当前本项目中最强可信 baseline 是：

```text
exp3_vote_5fold
DualBranchEEGNet Wide
BA = 0.6329 ± 0.0191
Macro-F1 = 0.6285 ± 0.0232
ROC-AUC = 0.6883 ± 0.0211
```

之前已经尝试过：

```text
DE/PSD/STFT8 + LogReg
DE/PSD/STFT8 + MLP
DE/PSD/STFT8 + Channel Graph Attention + Temporal Attention
ST-DADGAT-lite no-MMD
```

已有结果显示：

```text
STF LogReg: BA ≈ 0.6238
STF MLP: BA ≈ 0.6227
ST-DADGAT-lite no-MMD: BA ≈ 0.6136
```

结论是：**在只使用本项目数据从零训练的情况下，DE/PSD/STFT8 + Graph/Transformer 没有超过 EEGNet baseline。**

但是 Graph/Transformer 这条路线在论文中通常依赖：

1. 更强的预训练；
2. 更多跨被试数据；
3. 更充分的 subject-invariant 表征学习；
4. 更稳定的时空频特征编码器。

因此本任务的目标是：

> 使用无需人工申请的外部 EEG 情绪数据，对 DE/PSD/STFT8 + Graph/Transformer encoder 做预训练，然后在本项目数据上 fine-tune，重新检验这条路线是否能超过 `exp3_vote_5fold`。

---

## 1. 核心目标

实现并评估：

```text
External EEG emotion data
    ↓
DE / PSD / STFT8 feature extraction
    ↓
Graph + Transformer encoder pretraining
    ↓
Transfer encoder weights
    ↓
Fine-tune on local EEG dataset with exp3_vote_5fold split
    ↓
Compare against exp3_vote_5fold
```

最终要回答：

1. 外部数据预训练是否让 DE/PSD/STFT8 + Graph/Transformer 超过 `exp3_vote_5fold`？
2. 提升来自 supervised pretraining，还是 self-supervised pretraining？
3. Graph/Transformer 是否仍然过拟合？
4. 预训练是否改善 fold 3/4 的崩溃问题？
5. 这条路线是否值得继续深入？

---

## 2. 严格安全与公平性约束

### 2.1 禁止事项

1. 不要修改或覆盖已有 `exp3_vote_5fold` 结果。
2. 不要把外部数据和本项目数据直接混合后一起做最终 supervised training。
3. 不要把本项目 validation/test fold 用于预训练、scaler fit、feature selection 或 model selection。
4. 不要使用 public test 数据训练、预训练、调参或 early stopping。
5. 不要用 public test 反馈选择外部数据集、模型结构或超参数。
6. 不要在 split 之前对本项目全数据 fit scaler / normalizer / PCA / feature selector。
7. 不要在外部数据和本项目数据之间泄露标签映射，例如把本项目测试 subject 信息用于外部预训练采样策略。
8. 不要把 transductive 结果混入 inductive leaderboard。
9. 不要只报告最佳 fold 或最佳 seed。
10. 不要因为某个外部数据集表现差就偷偷丢掉不报告。

### 2.2 必须遵守

1. 最终 fine-tune 必须复用 `exp3_vote_5fold` 的 5-fold subject split。
2. 最终评估只在本项目数据上完成。
3. 每个 fold 内 scaler 只 fit 本项目 training fold。
4. 外部数据 scaler 和本项目 scaler 必须分开处理，不能把本项目 val/test 统计量带入预训练。
5. 预训练 encoder 可以迁移，classifier head 必须重新初始化。
6. 每个实验必须保存 config、split hash、metrics、predictions、leakage check。
7. 所有新结果放到：

```text
outputs/external_pretrain_graph_transformer/
```

8. 最终必须生成完整报告。
9. 如果外部数据下载失败，必须记录原因，不要伪造结果。
10. 如果标签无法统一，优先做 self-supervised pretraining，不要硬映射错误标签。

---

## 3. 推荐外部数据集

请优先寻找并接入 **无需人工审批申请即可下载** 的 EEG 情绪数据。

### 3.1 优先级 A：EEG Dataset for Emotion Classification Using Low-Cost and High-End Equipment

优先原因：

- EEG 情绪分类任务直接相关；
- 有 valence-arousal 四象限标签；
- 有 raw / preprocessed / extracted features；
- 设备包含高端 EEG 与低成本设备；
- 比较适合做 supervised auxiliary pretraining。

任务要求：

1. 自动检查数据是否可直接下载；
2. 记录 license；
3. 记录被试数；
4. 记录通道数；
5. 记录采样率；
6. 记录标签体系；
7. 判断是否能转换成四分类或二分类辅助任务；
8. 判断是否能抽取 DE/PSD/STFT8。

### 3.2 优先级 B：DENS / EEG Dataset on Emotion with Naturalistic Stimuli

优先原因：

- 被试数量较多；
- 自然主义情绪刺激；
- 可用于提升 cross-subject robustness；
- 适合 self-supervised 或 weak-supervised pretraining。

任务要求：

1. 检查 OpenNeuro 数据版本；
2. 记录 BIDS 结构；
3. 检查 EEG 通道；
4. 检查 label 文件；
5. 判断 valence/arousal/emotion category 是否可用；
6. 如果标签复杂，则优先做 self-supervised pretraining。

### 3.3 优先级 C：Voice-User Interaction Emotion EEG Dataset

优先原因：

- 公开情绪诱发 EEG 数据；
- 有 raw/preprocessed 数据；
- 任务评估中涉及 balanced accuracy / F1，和本项目较接近。

任务要求：

1. 检查下载方式；
2. 检查 `.mat` 数据结构；
3. 记录通道数、采样率、标签；
4. 判断是否适合作为辅助 supervised pretraining。

### 3.4 低优先级补充：FEEL Dataset

只作为补充，不作为主力预训练数据。

---

## 4. 不优先使用的数据集

以下数据集通常需要申请、EULA、机构邮箱或人工授权，不作为本任务首选：

```text
DREAMER
DEAP
SEED
SEED-IV
MAHNOB-HCI
ASCERTAIN
EEGEMO
EAV
```

如果 AI 找到这些数据集的公开镜像，也不要默认使用，必须先检查 license 和合规性。

---

## 5. 总体实验路线

本任务分成四个阶段：

```text
Stage 0: 外部数据可用性与格式审计
Stage 1: 外部数据 DE/PSD/STFT8 特征抽取
Stage 2: Graph/Transformer encoder 外部预训练
Stage 3: 本项目 exp3 split fine-tune
Stage 4: 对比、消融、报告
```

---

# Stage 0：外部数据审计

## 0.1 输出文件

生成：

```text
outputs/external_pretrain_graph_transformer/data_audit/external_dataset_inventory.csv
outputs/external_pretrain_graph_transformer/data_audit/external_dataset_audit.md
```

### `external_dataset_inventory.csv` 字段

```csv
dataset_name,source_url,download_status,license,subjects,channels,sampling_rate,labels,raw_available,preprocessed_available,feature_available,recommended_use,notes
```

### `external_dataset_audit.md` 内容

1. 每个候选数据集是否可下载；
2. 是否需要人工申请；
3. 是否允许科研使用；
4. 数据格式；
5. 标签定义；
6. 和本项目标签是否可对齐；
7. 是否适合 supervised pretraining；
8. 是否适合 self-supervised pretraining；
9. 最终选择哪些数据集进入 Stage 1。

## 0.2 数据集选择规则

优先选择：

```text
无需申请
license 清楚
EEG 情绪任务直接相关
被试数 >= 15
标签或 trial metadata 可解析
能提取 DE/PSD/STFT8
```

不满足这些条件的数据集可以跳过，但必须在审计报告中说明原因。

---

# Stage 1：外部数据特征抽取

## 1.1 目标

将不同外部数据集统一转换成：

```text
X_ext: [N, C_common, T=8, F]
y_ext: optional labels
subject_id_ext
dataset_id_ext
sample_id_ext
```

其中：

```text
C_common = 与本项目尽量对齐的通道集合
T = 8 STFT time bins
F = DE/PSD/STFT feature dim
```

## 1.2 通道对齐

如果外部数据通道数与本项目不同，按以下优先级处理：

### 方式 A：公共通道子集

优先选取本项目与外部数据都存在的通道，例如：

```text
Fp1, Fp2, F3, F4, F7, F8, C3, C4, P3, P4, O1, O2, T7, T8, Fz, Cz, Pz
```

实际通道列表必须由代码自动检查，不能硬猜。

### 方式 B：区域聚合

如果通道无法完全对齐，则将通道聚合为脑区：

```text
frontal
central
temporal
parietal
occipital
left
right
midline
```

得到 region-level features。

### 方式 C：channel interpolation

只有在 A/B 不可行时才使用插值，并在报告中说明。

## 1.3 采样率对齐

统一重采样到本项目使用的采样率，或统一到：

```text
128 Hz 或 256 Hz
```

具体选择以本项目原始 pipeline 为准。

必须记录：

```text
原始采样率
目标采样率
滤波范围
重采样方法
```

## 1.4 滤波与预处理

推荐：

```text
bandpass: 1-45 Hz
notch: 50 Hz 或 60 Hz，视数据来源而定
artifact handling: 使用数据集已有预处理优先
```

如果使用数据集 preprocessed 数据，不要重复做冲突处理。

## 1.5 特征定义

### DE

对经典频带计算 differential entropy：

```text
delta: 1-4 Hz
theta: 4-8 Hz
alpha: 8-13 Hz
beta: 13-30 Hz
gamma: 30-45 Hz
```

### PSD

同样按频带计算 band power 或 log band power。

### STFT8

将每个 trial/window 切成 8 个 time bins，计算每个 bin 内的频带能量或 log power。

最终建议：

```text
per channel per time bin:
[DE_delta..gamma, PSD_delta..gamma, STFT_bandpower_delta..gamma]
```

如果 DE/PSD 是 trial-level 静态特征，可复制到每个 time bin，与 STFT8 拼接。

## 1.6 标准化

外部预训练阶段：

```python
external_scaler.fit(external_train_data)
```

本项目 fine-tune 阶段：

```python
local_scaler.fit(local_train_fold)
```

禁止：

```python
scaler.fit(external + local all data)
scaler.fit(local all folds)
scaler.fit(local train + local val/test)
```

## 1.7 输出文件

```text
outputs/external_pretrain_graph_transformer/features/
├── external_features_{dataset}.npz
├── external_feature_metadata_{dataset}.csv
├── local_features_exp3_split_fold{fold}.npz
└── feature_extraction_report.md
```

metadata 至少包含：

```csv
sample_id,dataset_id,subject_id,trial_id,window_id,label,valence,arousal,emotion,split
```

---

# Stage 2：Graph/Transformer Encoder 外部预训练

## 2.1 模型结构

实现一个可迁移 encoder：

```text
Input [B, C, T, F]
    ↓
Feature Projection
    ↓
Channel Graph Attention
    ↓
Temporal Transformer
    ↓
Representation z
```

文件建议：

```text
models/graph_transformer_stf_encoder.py
models/channel_graph_attention.py
models/temporal_transformer.py
models/pretraining_heads.py
training/pretrain_external_graph_transformer.py
training/finetune_local_graph_transformer.py
```

## 2.2 Encoder 结构

### Feature Projection

```python
Linear(F, d_model)
LayerNorm
GELU
Dropout
```

推荐：

```yaml
d_model: 64
dropout: 0.3
```

### Channel Graph Attention

对每个 time bin 的 channel nodes 做 self-attention：

```text
H_t: [B, C, d_model]
GraphAttention(H_t) -> [B, C, d_model]
```

推荐实现：

```text
multi-head attention over channels
learnable adjacency bias A_bias[C, C]
residual connection
LayerNorm
dropout
```

### Temporal Transformer

先做 channel pooling 或保留 channel-token：

方案 A，简单稳定：

```text
H_graph: [B, C, T, d]
mean over channels -> [B, T, d]
Temporal Transformer -> [B, T, d]
mean over time -> z
```

方案 B，更复杂：

```text
flatten channel-time tokens -> [B, C*T, d]
Transformer Encoder -> [B, C*T, d]
pool -> z
```

优先用方案 A，避免过拟合。

推荐：

```yaml
graph_heads: 4
temporal_heads: 4
temporal_layers: 1
ffn_dim: 128
dropout: 0.3
```

## 2.3 预训练任务

至少实现两类预训练：

---

## Task A：Supervised External Emotion Pretraining

适用条件：

```text
外部数据有明确 emotion / valence / arousal 标签
```

训练：

```text
external X -> encoder -> external classifier head
```

标签映射优先级：

### 四象限标签

如果有 valence-arousal 四象限：

```text
HVHA
HVLA
LVHA
LVLA
```

直接做 4-class classification。

### 二分类标签

如果只有 valence 或 arousal：

```text
high/low valence
high/low arousal
```

可以做 multi-task binary classification：

```text
loss = CE_valence + CE_arousal
```

### 多类别 emotion

如果是 discrete emotion category：

```text
happy/sad/fear/neutral/...
```

做 dataset-specific classifier head，不强行映射到本项目标签。

注意：

预训练完成后，丢弃 external classifier head，只迁移 encoder。

---

## Task B：Self-supervised Masked Time-Bin Modeling

适用条件：

```text
任何外部 EEG 数据，即使标签不统一也可用
```

方法：

1. 随机 mask 部分 time bins；
2. encoder 根据未 mask 部分重建 masked feature；
3. 使用 MSE loss。

输入：

```text
X: [B, C, T=8, F]
mask: random time bins, ratio=0.25
```

loss：

```python
loss_recon = mse(pred_masked_features, target_masked_features)
```

---

## Task C：Subject-Invariant Contrastive Pretraining

目标：

```text
同一 emotion / 相似 valence-arousal 的样本拉近
不同 subject 但相似标签的样本拉近
避免 representation 只记住 subject identity
```

如果外部标签可靠：

```text
positive pairs:
same label, different subject

negative pairs:
different label
```

如果标签不可靠：

```text
positive pairs:
same trial 的不同 augmentation / time crop

negative pairs:
different trials
```

推荐先实现 supervised contrastive：

```python
loss_supcon = supervised_contrastive_loss(z, labels)
```

如果 batch 中 subject 足够多，加 subject decorrelation penalty：

```text
minimize emotion loss
maximize subject confusion / reduce subject predictability
```

但不要优先做 DANN，先做简单 contrastive。

---

## 2.4 预训练 loss 组合

建议先跑三个版本：

### P1：Supervised only

```text
loss = CE_external
```

### P2：Self-supervised only

```text
loss = reconstruction_loss
```

### P3：Supervised + contrastive

```text
loss = CE_external + lambda_contrastive * SupCon
```

推荐：

```yaml
lambda_contrastive: 0.1
```

不要一开始加入 MMD、DANN、复杂 adversarial loss。

---

## 2.5 预训练配置

```yaml
pretrain:
  seed: 42
  batch_size: 64
  epochs: 100
  optimizer: AdamW
  lr: 0.0005
  weight_decay: 0.01
  scheduler: cosine
  early_stopping: true
  patience: 15
  monitor: external_val_loss
  gradient_clip_norm: 1.0

model:
  d_model: 64
  graph_heads: 4
  temporal_heads: 4
  temporal_layers: 1
  ffn_dim: 128
  dropout: 0.3
  adjacency_bias: true
```

外部数据也要 subject-level split：

```text
external train subjects
external val subjects
```

禁止同一外部 subject 同时出现在 external train 和 external val。

---

## 2.6 预训练输出

每个预训练版本输出：

```text
outputs/external_pretrain_graph_transformer/pretrain/{pretrain_name}/
├── config.yaml
├── encoder_best.pt
├── external_classifier_head.pt
├── pretrain_metrics.json
├── pretrain_log.csv
├── external_split.json
├── external_leakage_check.json
└── pretrain_report.md
```

---

# Stage 3：本项目 Fine-tune

## 3.1 Fine-tune 原则

对每个 fold：

1. 读取 `exp3_vote_5fold` 的 train/val split；
2. 只用 local train fold fit scaler；
3. 加载 external pretrained encoder；
4. 丢弃 external classifier head；
5. 重新初始化 local classifier head；
6. 在 local train fold 上 fine-tune；
7. 用 local val/test fold 评估；
8. 保存 metrics 和 predictions。

## 3.2 Fine-tune 模式

必须比较至少三种：

### F0：From scratch Graph/Transformer

```text
DE/PSD/STFT8 + Graph/Transformer
随机初始化
```

用途：确认没有预训练时的性能。

### F1：Frozen encoder linear probe

```text
加载 external pretrained encoder
冻结 encoder
只训练 classifier head
```

用途：判断外部预训练 representation 是否可迁移。

### F2：Partial fine-tune

```text
加载 external pretrained encoder
冻结前若干层
微调 temporal transformer + classifier
```

用途：降低过拟合。

### F3：Full fine-tune

```text
加载 external pretrained encoder
全模型微调
```

用途：测试最大适配能力。

---

## 3.3 Fine-tune 配置

```yaml
finetune:
  seed: 42
  batch_size: 32
  epochs: 80
  optimizer: AdamW
  lr_encoder: 0.0001
  lr_head: 0.0005
  weight_decay: 0.01
  scheduler: cosine
  early_stopping: true
  patience: 15
  monitor: val_balanced_accuracy
  gradient_clip_norm: 1.0
  class_weight: auto_train_fold_only
```

注意：

```text
class_weight 只能从 local train fold 统计
```

不能用全数据统计类别权重。

## 3.4 Fine-tune 输出

```text
outputs/external_pretrain_graph_transformer/finetune/{pretrain_name}/{finetune_mode}/fold{fold}/
├── config.yaml
├── best_model.pt
├── metrics.json
├── predictions.csv
├── training_log.csv
├── split_integrity.json
└── leakage_check.json
```

---

# Stage 4：必须跑的实验矩阵

## 4.1 Baseline

读取已有：

```text
B0: exp3_vote_5fold
BA = 0.6329 ± 0.0191
MF1 = 0.6285 ± 0.0232
AUC = 0.6883 ± 0.0211
```

## 4.2 STF baseline

读取或复现：

```text
B1: DE+PSD+STFT8 + LogReg
BA ≈ 0.6238
```

## 4.3 Graph/Transformer from scratch

```text
G0: GraphTransformer-STF from scratch
```

## 4.4 External supervised pretraining

```text
G1: External supervised pretrain + linear probe
G2: External supervised pretrain + partial fine-tune
G3: External supervised pretrain + full fine-tune
```

## 4.5 External self-supervised pretraining

```text
G4: Masked time-bin pretrain + linear probe
G5: Masked time-bin pretrain + partial fine-tune
G6: Masked time-bin pretrain + full fine-tune
```

## 4.6 External supervised contrastive pretraining

```text
G7: SupCon pretrain + linear probe
G8: SupCon pretrain + partial fine-tune
G9: SupCon pretrain + full fine-tune
```

## 4.7 Optional：Multi-dataset pretraining

只有在单数据集预训练跑通后再做：

```text
G10: Multi-external supervised/self-supervised pretrain + best fine-tune mode
```

---

# Stage 5：成功标准

要进入主线候选，必须满足：

```text
BA >= 0.6379
```

或：

```text
Macro-F1 >= 0.6335
```

并且：

1. 5-fold CV；
2. 至少 3/5 folds 相比 `exp3_vote_5fold` 提升；
3. std 不明显高于 baseline；
4. 无 subject/trial/window 泄露；
5. no public usage；
6. no validation/test leakage；
7. 使用相同 split；
8. 外部数据只用于预训练，不参与 local final evaluation；
9. classifier head 在 local fine-tune 时重新初始化。

如果没有达到该标准，不能替代 `exp3_vote_5fold`。

---

# Stage 6：重点诊断 fold 3/4

之前 Graph/Transformer from scratch 在 fold 3/4 出现系统性崩溃。

本任务必须专门诊断：

1. 外部预训练是否改善 fold 3/4？
2. fold 3/4 的 subject 是否存在明显 domain shift？
3. 这些 subject 的 DE/PSD/STFT8 分布是否偏离 train subjects？
4. 错误主要集中在哪些类别？
5. linear probe、partial fine-tune、full fine-tune 哪个最稳？
6. 是否只有 full fine-tune 过拟合，linear probe 更稳？

输出：

```text
outputs/external_pretrain_graph_transformer/fold_diagnostics/fold34_diagnostics.md
outputs/external_pretrain_graph_transformer/fold_diagnostics/fold34_error_analysis.csv
```

字段：

```csv
fold,subject_id,class,n_samples,ba,mf1,main_confusion,feature_shift_score,notes
```

---

# Stage 7：泄露审计

必须生成：

```text
outputs/external_pretrain_graph_transformer/audit/pretrain_finetune_leakage_audit.md
outputs/external_pretrain_graph_transformer/audit/split_integrity.csv
outputs/external_pretrain_graph_transformer/audit/scaler_integrity.csv
outputs/external_pretrain_graph_transformer/audit/external_data_usage.csv
```

检查：

1. external pretraining subject split；
2. local fine-tune subject split；
3. local train/val/test subject overlap；
4. trial/window overlap；
5. scaler fit scope；
6. external labels 是否被错误映射；
7. local val/test 是否参与预训练；
8. public 是否参与任何训练；
9. external classifier head 是否被错误保留；
10. best checkpoint 是否基于 local validation，而不是 test/public。

---

# Stage 8：最终报告

生成：

```text
outputs/external_pretrain_graph_transformer/final_report/external_pretrain_graph_transformer_report.md
outputs/external_pretrain_graph_transformer/final_report/external_pretrain_graph_transformer_leaderboard.csv
outputs/external_pretrain_graph_transformer/final_report/pretraining_ablation.csv
outputs/external_pretrain_graph_transformer/final_report/finetune_mode_ablation.csv
outputs/external_pretrain_graph_transformer/final_report/vs_exp3_vote_5fold.csv
```

## 8.1 `external_pretrain_graph_transformer_report.md` 必须包含

```markdown
# External Pretraining for DE/PSD/STFT8 + Graph/Transformer

## 1. Motivation

## 2. External datasets
| Dataset | Subjects | Channels | Labels | License | Used for | Notes |

## 3. Feature extraction
- DE
- PSD
- STFT8
- channel alignment
- scaler policy

## 4. Model
- Graph attention
- Temporal transformer
- pretraining heads
- fine-tune classifier

## 5. Pretraining results
| Pretrain | External Dataset | Loss | Val Metric | Notes |

## 6. Local fine-tune results
| Experiment | Fine-tune Mode | BA | MF1 | AUC | ΔBA vs exp3 | ΔMF1 vs exp3 | Folds improved |

## 7. Per-fold comparison
| Fold | exp3 BA | Best Pretrained GraphTransformer BA | Delta | Notes |

## 8. Fold 3/4 diagnostics

## 9. Leakage audit

## 10. Final decision
- PROMOTE / SUPPLEMENT / REJECT
```

---

# Stage 9：最终终端摘要

任务完成后打印：

```text
===== External Pretraining Graph/Transformer Summary =====

Baseline exp3_vote_5fold:
  BA:
  Macro-F1:
  ROC-AUC:

Best external-pretrained Graph/Transformer:
  Experiment:
  External dataset:
  Pretraining type:
  Fine-tune mode:
  BA:
  Macro-F1:
  ROC-AUC:
  Delta BA:
  Delta Macro-F1:
  Folds improved:

Fold 3/4 improved:
  Fold 3 delta:
  Fold 4 delta:

Leakage:
  external split clean:
  local split clean:
  scaler clean:
  public used:
  local val/test used in pretrain:

Decision:
  PROMOTE / SUPPLEMENT / REJECT

Report:
  outputs/external_pretrain_graph_transformer/final_report/external_pretrain_graph_transformer_report.md
==============================================
```

---

# Stage 10：决策规则

## 情况 A：Promote

条件：

```text
BA >= 0.6379
MF1 >= 0.6335
>= 3/5 folds improved
std not significantly increased
no leakage
```

结论：

```text
External-pretrained Graph/Transformer becomes new main candidate.
```

## 情况 B：Supplement

条件：

```text
0.625 <= BA < 0.6379
```

或：

```text
fold 3/4 明显改善，但总体均值仍略低
```

结论：

```text
Keep exp3_vote_5fold as primary model.
Use external-pretrained Graph/Transformer as supplementary model or ensemble candidate.
```

## 情况 C：Reject

条件：

```text
BA < 0.625
```

或：

```text
std 明显增大
fold 3/4 仍崩
complex model still worse than LogReg
```

结论：

```text
Reject DE/PSD/STFT8 + Graph/Transformer for current project.
Return to exp3_vote_5fold and try backbone-level contrastive learning instead.
```

---

# Stage 11：如果时间有限，最小闭环

如果资源有限，只做以下最小版本：

```text
1. 接入一个外部数据集
2. 抽取 DE/PSD/STFT8
3. 训练 GraphTransformer encoder with supervised external labels
4. 在本项目上做：
   - linear probe
   - partial fine-tune
5. 与 exp3_vote_5fold 比较
6. 生成报告
```

最小实验矩阵：

```text
B0 exp3_vote_5fold
B1 STF LogReg
G0 GraphTransformer from scratch
G1 External supervised pretrain + linear probe
G2 External supervised pretrain + partial fine-tune
```

如果 G1/G2 都低于 0.625，则不要继续扩展。

---

# Stage 12：注意事项

1. 这次任务的目标不是证明 Graph/Transformer 一定强，而是公平地给它一次“外部数据预训练”的机会。
2. 如果仍然打不过 EEGNet，说明当前数据规模、特征表达或模型复杂度不适合这条路线。
3. 如果只在 1/5 fold 提升，不能算成功。
4. 如果均值略升但 std 大幅增加，也不能算稳定提升。
5. 如果外部预训练改善 fold 3/4，即使总 BA 没超过，也值得记录为有价值发现。
6. 如果 external supervised pretraining 不行，但 self-supervised 有改善，后续可以转向 masked EEG pretraining。
7. 如果 Graph/Transformer 还是不如 LogReg，不要再堆层数，先检查特征和 split。
8. 不要拿文献数字直接和本项目数字比较，必须只和本项目 `exp3_vote_5fold` 公平比较。
