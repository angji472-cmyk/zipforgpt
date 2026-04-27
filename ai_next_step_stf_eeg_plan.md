# AI 下一步任务：EEG 情绪分类的 DE/PSD/时频特征 + 空间-时间-频率编码器 + Graph/Transformer/Attention + Domain Adaptation/Contrastive Learning

## 0. 背景与当前结论

当前已有实验结论：

1. 当前可信最优基线是：

```text
exp3_vote_5fold
BA = 0.6329 ± 0.0191
Macro-F1 = 0.6285 ± 0.0232
ROC-AUC = 0.6883 ± 0.0211
```

2. AdaBN 对齐后确认无效：
   - R1 AdaBN train-only 在正确 Wide EEGNet baseline 上下降：
     - BA: 0.6329 → 0.6088
     - Macro-F1: 0.6285 → 0.5892
   - R2 AdaBN val-transductive 也下降：
     - BA: 0.6329 → 0.5817
     - Macro-F1: 0.6285 → 0.5346
   - 因此不要继续 AdaBN/DANN 主线。

3. 下一步转向更有结构性的特征与模型：

```text
DE / PSD / 时频特征
        ↓
spatial-temporal-frequency encoder
        ↓
graph / transformer / attention
        ↓
domain adaptation 或 contrastive learning
```

本任务目标是：在不破坏当前可信基线的前提下，构建一条新的 EEG 情绪分类研究线，验证手工频域/时频特征与现代编码器是否能稳定超过 `exp3_vote_5fold`。

---

## 1. 总体原则

### 1.1 不要做的事

1. 不要删除任何已有实验结果。
2. 不要覆盖 `EEGNET/outputs/exp3_vote_5fold`。
3. 不要继续 AdaBN。
4. 不要继续 DANN，除非前面的特征编码器已经稳定超过 baseline。
5. 不要一开始就上很复杂的大模型。
6. 不要把 public/test 数据用于训练、调参、early stopping、归一化统计或模型选择。
7. 不要在全数据上 fit scaler、PCA、特征选择器后再划分。
8. 不要把 transductive 结果混入 inductive leaderboard。
9. 不要只看 single-fold 结果得结论。
10. 不要使用 validation/test label 做任何 adaptation 或 contrastive mining。

### 1.2 必须做的事

1. 所有实验用 5-fold CV。
2. 尽量复用当前 trusted leaderboard 的 split。
3. 所有新输出放到新目录：

```text
EEGNET/outputs_stf_experiments/
```

4. 每个实验保存：
   - config snapshot
   - run command
   - random seed
   - fold split hash
   - train/val subject list
   - feature config
   - model config
   - summary_metrics.json
   - fold_metrics.csv
   - predictions.csv
   - run_metadata.json
   - leakage_check.json

5. 每个实验必须和当前最优基线比较：

```text
baseline = exp3_vote_5fold
BA = 0.6329 ± 0.0191
Macro-F1 = 0.6285 ± 0.0232
ROC-AUC = 0.6883 ± 0.0211
```

6. 主排序指标：

```text
balanced_accuracy mean
```

辅助指标：

```text
macro_f1 mean
roc_auc mean
std
per-fold delta
```

---

## 2. 项目结构建议

请新建以下目录，不要污染旧代码：

```text
EEGNET/stf_features/
├── extract_de.py
├── extract_psd.py
├── extract_timefreq.py
├── feature_cache.py
├── feature_normalization.py
└── feature_quality_check.py

EEGNET/stf_models/
├── stf_dataset.py
├── stf_encoder.py
├── graph_encoder.py
├── transformer_encoder.py
├── attention_pooling.py
├── contrastive_heads.py
└── domain_heads.py

EEGNET/scripts_stf/
├── build_stf_features.py
├── train_stf_baselines.py
├── train_stf_graph.py
├── train_stf_transformer.py
├── train_stf_contrastive.py
├── evaluate_stf.py
└── audit_stf_leakage.py

EEGNET/configs_stf/
├── feature_de_psd.yaml
├── feature_timefreq.yaml
├── model_mlp_baseline.yaml
├── model_stf_attention.yaml
├── model_graph_gat.yaml
├── model_transformer.yaml
├── model_contrastive.yaml
└── experiment_grid.yaml

EEGNET/outputs_stf_experiments/
├── feature_cache/
├── reports/
├── leaderboards/
└── runs/
```

---

## 3. 阶段总览

请严格按以下阶段执行：

| 阶段 | 目标 | 是否必须完成 |
|---|---|---|
| Stage 0 | 复用 split，建立审计框架 | 必须 |
| Stage 1 | 提取 DE / PSD / 时频特征 | 必须 |
| Stage 2 | 轻量 baseline：LogReg / SVM / MLP | 必须 |
| Stage 3 | spatial-temporal-frequency encoder | 必须 |
| Stage 4 | Graph / Transformer / Attention | 视 Stage 3 结果 |
| Stage 5 | Contrastive learning | 视 Stage 4 结果 |
| Stage 6 | Domain adaptation | 最后再做 |
| Stage 7 | 总结报告与是否替代 exp3_vote_5fold | 必须 |

---

## 4. Stage 0：Split 与审计框架

### 4.1 目标

确保新实验与旧实验可公平比较。

### 4.2 任务

1. 从当前可信实验中提取 split：

```text
EEGNET/outputs/exp3_vote_5fold
```

2. 生成统一 split 文件：

```text
EEGNET/outputs_stf_experiments/splits/exp3_vote_5fold_splits.json
```

每个 fold 包含：

```json
{
  "fold": 0,
  "train_subjects": [],
  "val_subjects": [],
  "test_subjects": [],
  "train_sample_ids": [],
  "val_sample_ids": [],
  "test_sample_ids": [],
  "split_hash": ""
}
```

3. 如果旧实验没有完整 sample_id，则从 prediction CSV、metadata、dataset 索引或训练日志恢复。

4. 如果无法恢复完整 split，则生成新的 GroupKFold split，但必须：
   - 按 subject 分组；
   - 同一个 subject 不能跨 train/val；
   - 固定 seed；
   - 保存 split_hash；
   - 在报告中说明“不是与旧实验完全配对比较”。

### 4.3 输出

```text
EEGNET/outputs_stf_experiments/reports/split_reuse_report.md
EEGNET/outputs_stf_experiments/splits/exp3_vote_5fold_splits.json
```

---

## 5. Stage 1：特征提取

### 5.1 总体输入输出

输入原始或预处理后的 EEG window/trial 数据，输出结构化特征：

```text
sample_id
subject_id
trial_id
window_id
label
fold
split
feature_tensor
```

推荐统一 tensor 格式：

```text
X_feature shape = [num_samples, num_channels, num_bands, num_time_bins, num_feature_types]
```

如果某些特征没有 time_bins，则设：

```text
num_time_bins = 1
```

---

## 6. DE 特征

### 6.1 Differential Entropy 定义

对每个 channel、每个 frequency band、每个时间窗，计算：

```text
DE = 0.5 * log(2 * pi * e * variance)
```

其中 variance 从对应频带滤波后的信号估计。

### 6.2 频带建议

先使用经典 EEG 频带：

```text
delta: 1-4 Hz
theta: 4-8 Hz
alpha: 8-13 Hz
beta: 13-30 Hz
gamma: 30-45 Hz
```

如果采样率或预处理不支持 gamma，则改为：

```text
gamma: 30-40 Hz
```

所有频带边界写入 config。

### 6.3 实现要求

1. 每个样本独立计算 DE。
2. 不使用 label。
3. 不在全数据上 fit 任何参数。
4. 滤波器参数固定。
5. 如果需要标准化，必须在每个 fold 内只用 train split fit。
6. 保存每个 feature config 的 hash。

### 6.4 输出

```text
EEGNET/outputs_stf_experiments/feature_cache/de_features.npz
EEGNET/outputs_stf_experiments/reports/de_feature_report.md
```

---

## 7. PSD / Bandpower 特征

### 7.1 PSD 方法

使用 Welch 方法或已有稳定 PSD 函数。

推荐配置：

```yaml
psd:
  method: welch
  nperseg: auto
  noverlap: 0.5
  window: hann
  bands:
    delta: [1, 4]
    theta: [4, 8]
    alpha: [8, 13]
    beta: [13, 30]
    gamma: [30, 45]
```

### 7.2 特征类型

至少计算：

```text
absolute_bandpower
relative_bandpower
log_bandpower
```

可选：

```text
bandpower_ratio
theta_alpha_ratio
beta_alpha_ratio
frontal_asymmetry
```

### 7.3 安全要求

1. PSD 本身逐样本计算，低泄露风险。
2. relative_bandpower 的 total power 必须来自该样本本身，不得用全数据统计。
3. 标准化必须 fold 内 train-only fit。
4. 不要先全局标准化再 split。

### 7.4 输出

```text
EEGNET/outputs_stf_experiments/feature_cache/psd_features.npz
EEGNET/outputs_stf_experiments/reports/psd_feature_report.md
```

---

## 8. 时频特征

### 8.1 推荐两种方案

先做轻量方案，后做重方案。

#### 方案 A：STFT

输出：

```text
X_stft = [channels, frequency_bins, time_bins]
```

然后按 EEG bands 聚合成：

```text
X_stft_band = [channels, bands, time_bins]
```

#### 方案 B：Morlet CWT

输出：

```text
X_cwt = [channels, scales/frequencies, time_bins]
```

再聚合到 bands。

### 8.2 不建议一开始做的方案

不要一开始把时频图当作大图丢给大型 CNN/ViT。样本量通常不够，容易过拟合。

### 8.3 推荐输出格式

```text
X_timefreq_band = [samples, channels, bands, time_bins, feature_types]
```

feature_types 可包含：

```text
power
log_power
phase_locking_optional
```

### 8.4 输出

```text
EEGNET/outputs_stf_experiments/feature_cache/timefreq_stft_features.npz
EEGNET/outputs_stf_experiments/feature_cache/timefreq_cwt_features.npz
EEGNET/outputs_stf_experiments/reports/timefreq_feature_report.md
```

---

## 9. Stage 1 质量检查

每个特征文件必须跑 quality check：

### 9.1 检查内容

1. shape 是否正确；
2. NaN / Inf 数量；
3. 每个 band 的均值和方差；
4. 每个 channel 的均值和方差；
5. 每个 split 的样本数；
6. 每个 split 的类别分布；
7. subject 是否跨 train/val；
8. trial/window 是否跨 train/val；
9. 特征是否包含 label 字段；
10. sample_id 是否唯一。

### 9.2 输出

```text
EEGNET/outputs_stf_experiments/reports/feature_quality_report.md
```

---

## 10. Stage 2：轻量特征 baseline

在上复杂模型前，必须先验证特征本身是否有信息量。

### 10.1 实验列表

#### E1：DE + Logistic Regression

```text
stf_e1_de_logreg_5fold
```

#### E2：PSD + Logistic Regression

```text
stf_e2_psd_logreg_5fold
```

#### E3：DE + PSD + Logistic Regression

```text
stf_e3_de_psd_logreg_5fold
```

#### E4：DE + PSD + SVM

```text
stf_e4_de_psd_svm_5fold
```

#### E5：DE + PSD + MLP

```text
stf_e5_de_psd_mlp_5fold
```

### 10.2 训练规则

1. 每个 fold 内：
   - fit scaler on train only；
   - transform train/val；
   - train classifier；
   - evaluate val。
2. 使用 GroupKFold split。
3. 不使用 public。
4. 不使用 test/public 调参。
5. 如果调 C、kernel、hidden dim，只能用 inner validation 或固定小网格，必须记录为 tuning bias。

### 10.3 进入下一阶段的标准

如果 E3/E4/E5 中至少一个达到：

```text
BA >= 0.60
Macro-F1 >= 0.59
```

则继续 Stage 3。

如果所有轻量 baseline 都明显低于 0.58，则说明特征管线可能有问题，先排查特征。

---

## 11. Stage 3：Spatial-Temporal-Frequency Encoder

### 11.1 目标

构建一个能同时编码：

```text
spatial/channel
temporal/time_bins
frequency/bands
feature_type/DE-PSD-STFT
```

的中等复杂度模型。

### 11.2 推荐输入

```text
X = [B, C, F, T, K]
```

含义：

```text
B = batch size
C = EEG channels
F = frequency bands
T = time bins
K = feature types，例如 DE / PSD / log_power
```

如果只有 DE/PSD：

```text
T = 1
```

### 11.3 编码器设计：STFEncoder-v1

#### 模块 1：Feature projection

```text
Linear(K -> d_model)
LayerNorm
GELU
Dropout
```

输出：

```text
[B, C, F, T, d_model]
```

#### 模块 2：Frequency encoder

对每个 channel、time bin，在 band 维度做 attention 或 MLP：

```text
BandAttention / BandMLP
```

输出仍为：

```text
[B, C, F, T, d_model]
```

#### 模块 3：Temporal encoder

对每个 channel、band，在时间维度做：

```text
TemporalConv1D
或
GRU
或
TransformerEncoder
```

如果 T=1，则跳过。

#### 模块 4：Spatial encoder

对 channel 维度做：

```text
ChannelAttention
或
GraphAttention
或
Transformer over channels
```

#### 模块 5：Pooling

比较三种 pooling：

```text
mean pooling
attention pooling
CLS token pooling
```

#### 模块 6：Classifier

```text
MLP(d_model -> hidden -> num_classes)
```

### 11.4 第一个模型不要太大

建议默认配置：

```yaml
model:
  name: STFEncoderV1
  d_model: 64
  n_heads: 4
  n_layers_freq: 1
  n_layers_time: 1
  n_layers_spatial: 1
  dropout: 0.3
  classifier_hidden: 64
  weight_decay: 1e-4
```

### 11.5 实验列表

#### E6：DE+PSD + STFEncoder

```text
stf_e6_de_psd_stfencoder_5fold
```

#### E7：DE+PSD+STFT + STFEncoder

```text
stf_e7_de_psd_stft_stfencoder_5fold
```

#### E8：DE+PSD+CWT + STFEncoder

```text
stf_e8_de_psd_cwt_stfencoder_5fold
```

### 11.6 进入下一阶段标准

如果 E6/E7/E8 相比轻量 baseline 提升：

```text
BA >= +0.005
或
Macro-F1 >= +0.005
```

则进入 Stage 4。

如果没有提升，先不要上 Transformer/Graph，先排查过拟合。

---

## 12. Stage 4：Graph / Transformer / Attention

### 12.1 Graph 建模思路

EEG channel 天然有空间拓扑。构图方式分三种：

#### Graph A：物理邻接图

根据电极空间距离构图：

```text
edge_weight = exp(-distance^2 / sigma^2)
```

如果没有电极坐标，则使用标准 10-20 系统近邻。

#### Graph B：功能连接图

基于 train split 计算：

```text
correlation
coherence
PLV
mutual information optional
```

注意：

> 功能连接图必须每个 fold 内只用 train 数据计算，不能用 val/test。

#### Graph C：可学习图

模型学习 adjacency，但需要强正则。

---

### 12.2 Graph 模型

#### E9：DE+PSD + GCN/GAT

```text
stf_e9_de_psd_gat_5fold
```

输入：

```text
node = channel
node_feature = flatten(frequency, time, feature_type)
```

模型：

```text
Channel feature projection
GATConv / GraphConv
Attention pooling
Classifier
```

默认配置：

```yaml
graph:
  type: physical
  layer: GAT
  hidden: 64
  heads: 4
  layers: 2
  dropout: 0.3
```

---

### 12.3 Transformer 模型

#### E10：STF tokens + Transformer

```text
stf_e10_de_psd_transformer_5fold
```

Token 定义：

```text
token = channel × band × time_bin
```

每个 token 加：

```text
channel embedding
band embedding
time embedding
feature projection
```

模型：

```text
TransformerEncoder
CLS token
MLP classifier
```

默认配置：

```yaml
transformer:
  d_model: 64
  n_heads: 4
  n_layers: 2
  dim_feedforward: 128
  dropout: 0.3
  use_cls_token: true
```

注意：

> 如果样本数不大，不要超过 2 层 Transformer，不要 d_model > 128。

---

### 12.4 Attention-only 模型

#### E11：Channel-Band Attention

```text
stf_e11_channel_band_attention_5fold
```

模型：

```text
Band attention
Channel attention
Gated pooling
Classifier
```

这个模型通常比完整 Transformer 更稳，建议作为主力候选。

---

## 13. Stage 5：Contrastive Learning

### 13.1 目标

通过自监督或监督对比学习增强 subject-invariant 表征。

注意：

> 对比学习必须非常谨慎，不能让同一 trial/window 的增强版本跨 train/val。

---

### 13.2 推荐先做 supervised contrastive

#### E12：STFEncoder + Supervised Contrastive

```text
stf_e12_stfencoder_supcon_5fold
```

训练 loss：

```text
loss = CE_loss + lambda_contrast * SupCon_loss
```

推荐：

```yaml
contrastive:
  enabled: true
  type: supervised
  lambda: 0.1
  temperature: 0.1
  projection_dim: 64
```

正样本：

```text
同一 train split 中同 label 的样本
```

负样本：

```text
train split 中不同 label 的样本
```

禁止：

```text
使用 val/test 样本构造正负对
```

---

### 13.3 自监督 contrastive 可选

#### E13：STFEncoder + SimCLR-style augmentation

```text
stf_e13_stfencoder_simclr_5fold
```

增强方式：

```text
Gaussian noise
channel dropout
frequency band dropout
time masking
amplitude scaling
```

禁止：

```text
同一原始 trial 的增强版本一部分进 train、一部分进 val
```

所有增强只在 train split 内执行。

---

### 13.4 进入主线标准

Contrastive 只有满足以下条件才保留：

1. BA 提升 >= 0.005；
2. Macro-F1 不下降；
3. std 不明显增大；
4. 至少 3/5 fold 提升；
5. 没有 subject/trial/window 泄露。

---

## 14. Stage 6：Domain Adaptation

Domain adaptation 最后再做，不要一开始做。

### 14.1 推荐方向：Subject-adversarial representation

如果要做，先不要用 val/test/public 作为 target。

#### E14：Subject-adversarial train-only

```text
stf_e14_subject_adversarial_train_only_5fold
```

目标：

```text
学到 emotion-discriminative 但 subject-invariant 的表示
```

结构：

```text
encoder -> emotion_classifier
        -> gradient_reversal -> subject_classifier
```

loss：

```text
loss = emotion_loss + lambda_subject * subject_adversarial_loss
```

注意：

- subject_classifier 只在 train split 内训练；
- val subject 不参与 adversarial 训练；
- 不使用 public；
- 这是 inductive setting。

---

### 14.2 CORAL / MMD 可选

#### E15：Train-only subject alignment

```text
stf_e15_mmd_coral_train_only_5fold
```

在 train subjects 之间做分布对齐。

不要使用 val/test/public。

---

### 14.3 不推荐当前做的事情

暂时不要做：

```text
val-transductive domain adaptation
public-transductive domain adaptation
DANN with validation target
DANN with public target
```

因为前面 AdaBN 已经证明 transductive adaptation 很容易破坏模型统计。

---

## 15. 训练配置统一要求

### 15.1 Optimizer

默认：

```yaml
optimizer:
  type: AdamW
  lr: 1e-3
  weight_decay: 1e-4
```

### 15.2 Scheduler

```yaml
scheduler:
  type: ReduceLROnPlateau
  mode: max
  monitor: val_balanced_accuracy
  factor: 0.5
  patience: 8
```

### 15.3 Early stopping

```yaml
early_stopping:
  monitor: val_balanced_accuracy
  patience: 20
  min_delta: 0.001
```

### 15.4 Batch size

```yaml
batch_size: 32
```

如果样本小：

```yaml
batch_size: 16
```

### 15.5 Loss

默认：

```yaml
loss:
  type: cross_entropy
```

可选：

```yaml
loss:
  type: focal
  gamma: 2.0
```

class weight 必须只从 train fold 计算。

---

## 16. 归一化规则

### 16.1 特征级标准化

对 DE/PSD/STFT 特征做标准化时：

```text
fit scaler on train fold only
transform train fold
transform val fold
```

禁止：

```text
fit scaler on all data before split
```

### 16.2 推荐标准化维度

建议比较两种：

#### Norm A：global feature scaler

对 flatten feature 做 StandardScaler。

#### Norm B：channel-band scaler

对每个 channel-band-feature 单独 fit mean/std。

优先使用 Norm B。

---

## 17. 防泄露检查

每个实验必须生成：

```text
leakage_check.json
```

格式：

```json
{
  "experiment": "",
  "setting": "inductive",
  "uses_public": false,
  "uses_val_for_training": false,
  "uses_val_for_adaptation": false,
  "uses_val_labels_for_training": false,
  "normalization_fit_scope": "train_only",
  "feature_fit_scope": "sample_independent_or_train_only",
  "subject_overlap": false,
  "trial_overlap": false,
  "window_overlap": false,
  "augmentation_train_only": true,
  "contrastive_pairs_train_only": true,
  "graph_fit_scope": "fixed_or_train_only",
  "risk_level": "low",
  "notes": ""
}
```

---

## 18. 实验执行顺序

请严格按以下顺序执行。

### Step 1：构建特征缓存

```text
build_de_features
build_psd_features
build_stft_features
feature_quality_check
```

### Step 2：轻量 baseline

```text
stf_e1_de_logreg_5fold
stf_e2_psd_logreg_5fold
stf_e3_de_psd_logreg_5fold
stf_e4_de_psd_svm_5fold
stf_e5_de_psd_mlp_5fold
```

### Step 3：STF encoder

```text
stf_e6_de_psd_stfencoder_5fold
stf_e7_de_psd_stft_stfencoder_5fold
```

### Step 4：Attention / Graph / Transformer

优先：

```text
stf_e11_channel_band_attention_5fold
```

然后：

```text
stf_e9_de_psd_gat_5fold
stf_e10_de_psd_transformer_5fold
```

### Step 5：Contrastive

只在 E6/E11 表现接近 baseline 时做：

```text
stf_e12_stfencoder_supcon_5fold
stf_e13_stfencoder_simclr_5fold
```

### Step 6：Domain adaptation

只在 contrastive/encoder 表现稳定后做：

```text
stf_e14_subject_adversarial_train_only_5fold
stf_e15_mmd_coral_train_only_5fold
```

---

## 19. 结果判定标准

### 19.1 可以进入主线

满足：

```text
BA >= baseline BA + 0.005
或
Macro-F1 >= baseline Macro-F1 + 0.005
```

也就是：

```text
BA >= 0.6379
或
Macro-F1 >= 0.6335
```

同时必须满足：

1. 5-fold CV；
2. inductive；
3. 无 subject/trial/window 泄露；
4. std 不明显增大；
5. 至少 3/5 folds 提升；
6. 没有 public/test 参与。

---

### 19.2 可以作为补充

满足：

```text
BA 接近 baseline，例如 0.625 ~ 0.637
Macro-F1 接近 baseline，例如 0.620 ~ 0.633
```

但没有稳定超过 baseline。

---

### 19.3 应该放弃

满足任意一个：

```text
BA < 0.61
Macro-F1 < 0.60
std 明显变大
只有 single-fold 提升
出现泄露风险
需要 transductive 才提升
```

---

## 20. 输出文件

最终生成：

```text
EEGNET/outputs_stf_experiments/reports/stf_master_report.md
EEGNET/outputs_stf_experiments/leaderboards/raw_stf_leaderboard.csv
EEGNET/outputs_stf_experiments/leaderboards/trusted_stf_leaderboard.csv
EEGNET/outputs_stf_experiments/leaderboards/comparison_against_exp3_vote.csv
EEGNET/outputs_stf_experiments/reports/feature_quality_report.md
EEGNET/outputs_stf_experiments/reports/leakage_audit_report.md
EEGNET/outputs_stf_experiments/reports/ablation_summary.md
```

每个实验目录：

```text
EEGNET/outputs_stf_experiments/runs/<experiment_name>/
├── config_snapshot.yaml
├── run_metadata.json
├── leakage_check.json
├── summary_metrics.json
├── fold_metrics.csv
├── predictions.csv
├── classification_report.csv
├── confusion_matrix.png
├── roc_curve.png
└── pr_curve.png
```

---

## 21. stf_master_report.md 必须回答的问题

1. DE 特征单独效果如何？
2. PSD 特征单独效果如何？
3. DE+PSD 是否优于单独特征？
4. STFT/CWT 时频特征是否带来增益？
5. 轻量模型中谁最好？
6. STFEncoder 是否超过轻量 baseline？
7. Graph/GAT 是否带来 channel 空间建模收益？
8. Transformer 是否过拟合？
9. Channel-band attention 是否比完整 Transformer 更稳？
10. Contrastive learning 是否有稳定收益？
11. Domain adaptation 是否值得继续？
12. 是否有任何模型超过 `exp3_vote_5fold`？
13. 如果没有，哪些模块仍值得保留？
14. 是否建议进入下一轮实验？

---

## 22. 终端最终摘要格式

最后输出：

```text
===== STF EEG Experiment Summary =====

Baseline:
  exp3_vote_5fold
  BA: 0.6329 ± 0.0191
  Macro-F1: 0.6285 ± 0.0232
  ROC-AUC: 0.6883 ± 0.0211

Feature baselines:
  best DE:
  best PSD:
  best DE+PSD:
  best time-frequency:

Encoder models:
  best STFEncoder:
  best Attention:
  best Graph:
  best Transformer:

Contrastive:
  best:
  improves baseline:

Domain adaptation:
  best:
  improves baseline:

Trusted best new model:
  name:
  BA:
  Macro-F1:
  ROC-AUC:
  delta BA vs baseline:
  delta Macro-F1 vs baseline:
  folds improved:
  risk level:

Recommendation:
  keep / reject / further test

Generated files:
  stf_master_report:
  trusted leaderboard:
====================================
```

---

## 23. 最重要的执行建议

优先级最高：

```text
DE+PSD feature baseline
↓
DE+PSD + Channel-Band Attention
↓
DE+PSD+STFT + STFEncoder
↓
SupCon
```

暂时不要优先做：

```text
大 Transformer
复杂 DANN
public transductive adaptation
大规模 CWT 图像 CNN
```

当前最可能有收益的方向是：

```text
频带能量/DE 特征 + 小型 channel-band attention + subject-level split + train-only contrastive
```

而不是继续 AdaBN/DANN。
