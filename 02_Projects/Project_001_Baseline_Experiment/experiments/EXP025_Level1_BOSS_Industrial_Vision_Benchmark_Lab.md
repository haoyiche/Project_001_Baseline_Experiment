# EXP025 — Level 1 BOSS
# Industrial Vision Benchmark Lab

---

# 1. 实验定位

## 当前 Level

```text
Level 1
Visual Foundations & Experiment Capability
视觉基础与实验能力
```

## 实验编号

```text
EXP025
```

## 实验主题

```text
Industrial Vision Benchmark Lab
工业视觉 Benchmark 实验室
```

## 当前状态

```text
EXP025     CONTINUE
Level 1    CONTINUE
```

---

# 2. 实验目标

本实验作为 Level 1 BOSS，目标不是单纯得到一个更高的指标，而是完成一次完整的工业视觉实验闭环。

核心任务：

```text
同一数据集
→ 两种不同方法范式
→ 统一评估协议
→ 指标比较
→ Failure Analysis
→ 主动修改
→ 原因验证
→ 工程结论
→ 可复现
```

最终需要证明自己具备以下能力：

```text
能建立 Baseline
能解释指标
能比较方法
能设计消融
能发现 Failure
能提出原因假设
能主动修改变量
能先预测后验证
能分析因果边界
能复现实验
能进行工程 Trade-off
```

---

# 3. 前置实验基础

EXP025 建立在 EXP008 ～ EXP024 的学习基础上。

主要前置能力包括：

```text
PatchCore / Feature Memory
Nearest Neighbor
AUROC / AP
Failure Case Analysis
Repeated Runs
Aggregation Ablation
Feature Layer Ablation
Latency Benchmark
Object Detection Evaluation
Embedding Geometry
Cosine / L2
Image Retrieval
Supervised ViT
DINOv3
Patch Token
Spatial Anomaly Localization
Resolution Ablation
Boundary FP Diagnosis
```

与 EXP024 最直接的连接：

```text
DINOv3 Patch Representation
+
Nearest Neighbor
+
Spatial Anomaly Map
```

已经在：

```text
MVTec AD
Bottle
broken_small
```

上完成 Pixel-level anomaly localization。

EXP025 在此基础上增加第二种真正不同的方法范式：

```text
Reconstruction-based Anomaly Detection
```

---

# 4. 数据集

使用：

```text
MVTec AD
Bottle
```

数据根目录：

```text
data/raw/mvtec_anomaly_detection/bottle/
```

正常训练集：

```text
train/good
209 images
```

主要异常测试集：

```text
test/broken_small
22 images
```

Ground Truth：

```text
ground_truth/broken_small
```

本轮 BOSS 的主要定位任务集中在：

```text
broken_small
```

原因：

```text
小缺陷
→ 对空间粒度、异常分数排序、False Positive
更加敏感
```

---

# 5. 两类方法

---

## 5.1 Method A：DINOv3 Feature + Nearest Neighbor

模型：

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

结构：

```text
ViT-S/16

Patch Size       = 16
Hidden Size      = 384
Register Tokens  = 4
```

224 输入时：

```text
224 / 16 = 14

Patch Grid:
14 × 14

Patch Tokens:
196

Sequence:
1 CLS
+ 4 Register
+ 196 Patch
= 201 Tokens
```

异常检测流程：

```text
正常训练图像
→ 提取 Patch Embedding
→ 建立 Normal Feature Memory

测试图像
→ 提取 Patch Embedding
→ 与 Normal Memory 最近邻匹配
→ 得到 Patch Distance
→ 作为 Anomaly Score
→ 上采样至原始图像空间
```

核心假设：

```text
正常 Patch
→ 在 Normal Feature Space 中存在相似邻居

异常 Patch
→ 与 Normal Memory 距离较大
```

---

## 5.2 Method B：Convolutional Autoencoder

方法类型：

```text
Reconstruction-based Anomaly Detection
```

基本流程：

```text
正常图像
→ Encoder
→ Bottleneck
→ Decoder
→ Reconstruction
```

测试时：

```text
输入 x
→ 重建 x_hat
→ 计算 |x - x_hat|
→ Pixel-level Anomaly Score
```

异常分数：

```text
score(i) =
mean RGB absolute reconstruction error
```

即：

```text
score = mean(|x - x_hat|, channel)
```

核心假设：

```text
正常区域
→ 模型能够较好重建
→ Reconstruction Error 小

异常区域
→ 模型没有见过
→ Reconstruction Error 大
```

---

# 6. Autoencoder 最小 Baseline

## 6.1 模型结构

Encoder：

```text
3
→ 32
→ 64
→ 128
→ 256
```

每层：

```text
Conv2d
kernel = 3
stride = 2
padding = 1
ReLU
```

空间尺寸：

```text
224
→ 112
→ 56
→ 28
→ 14
```

因此 Bottleneck：

```text
256 × 14 × 14
```

Decoder：

```text
256
→ 128
→ 64
→ 32
→ 3
```

使用：

```text
ConvTranspose2d
+
ReLU
```

最后：

```text
Sigmoid
```

---

## 6.2 训练参数

```text
Seed          = 42
Input         = 224 × 224
Batch Size    = 8
Epochs        = 20
Optimizer     = Adam
Learning Rate = 1e-3
Loss          = L1
```

训练数据：

```text
209 normal Bottle images
```

---

# 7. 实验前预测

在运行 Autoencoder 前，先进行了以下预测。

---

## Prediction 1

DINOv3 Patch-NN 预计比基础 Autoencoder 更适合 small defect localization。

原因：

```text
DINOv3
→ pretrained local representation
→ Patch-level nearest-neighbor comparison

Autoencoder
→ reconstruction
→ 容易平滑掉细节
→ 正常高频结构本身也可能难重建
```

注意：

```text
DINOv3 Patch Token
不等于天然具有高空间分辨率
```

224 输入时仍然只有：

```text
14 × 14 Patch Grid
```

---

## Prediction 2

Autoencoder 容量太大可能反而不利于 anomaly detection。

原因：

```text
容量增加
→ 正常图像重建更好

但同时
→ 异常区域也可能重建更好
```

因此：

```text
Lower Reconstruction Loss
≠
Better Anomaly Detection
```

---

## Prediction 3

基础 Pixel Anomaly Score 可以使用：

```text
|x_i - x_hat_i|
```

RGB 图像中：

```python
torch.abs(x - x_hat).mean(dim=1)
```

也可以考虑：

```text
L2
SSIM
Perceptual Feature
```

但本实验首先使用最小 L1 Pixel Reconstruction Error。

---

## Prediction 4

如果：

```text
AUROC 尚可
但 AP 很低
```

可能意味着：

```text
存在大量高分 False Positive

+
Pixel-level 类别极度不平衡

+
异常区域较小
```

因此：

```text
High-score Negative Tail
```

可能是重点 Failure Mechanism。

---

# 8. 第一次 Autoencoder Baseline

第一次实验：

```text
Input = 224
Epoch = 20
```

得到：

```text
Mean Pixel AUROC:
0.779785

Mean Pixel AP:
0.181338

Global Pixel AUROC:
0.774578

Global Pixel AP:
0.175340
```

初步观察：

```text
GT+ Mean
普遍高于
GT- Mean
```

说明：

```text
Autoencoder 并不是完全没有异常信号
```

但是 AP 很低，说明：

```text
异常区域虽然平均得分更高

但大量 Normal Pixel
仍然进入高分 Ranking
```

---

# 9. Evaluation Protocol 问题发现

初始实验存在一个重要问题。

DINOv3：

```text
14 × 14 anomaly map
→ 上采样到原图
→ 与原始 GT 比较
```

而初始 Autoencoder：

```text
224 × 224 anomaly map
→ GT resize 到 224
→ 在 224 空间比较
```

因此：

```text
两种方法 Evaluation Geometry 不一致
```

不能直接进行严格对比。

---

# 10. Common Evaluation Protocol Repair

决定统一为：

```text
模型内部可以使用自己的输入尺寸

但最终 anomaly score
必须映射回原始图像空间

Ground Truth 保持原始尺寸
```

Autoencoder：

```text
224 × 224 score map
→ bilinear interpolate
→ original H × W
```

Ground Truth：

```text
保持原始尺寸
不 resize
```

最终：

```text
score_map.shape == gt.shape
```

---

# 11. Protocol Repair Debug

修改过程中出现错误：

```text
UnboundLocalError:
local variable 'score_map'
referenced before assignment
```

原因：

在修改 evaluation protocol 时：

```text
原来的 reconstruction error 计算被删掉
```

但仍然保留：

```python
score_map = score_map.unsqueeze(1)
```

导致：

```text
score_map 尚未定义
```

正确数据流：

```text
x
[1, 3, 224, 224]

x_hat
[1, 3, 224, 224]

abs(x - x_hat)
[1, 3, 224, 224]

mean(channel)
[1, 1, 224, 224]

interpolate
[1, 1, H_original, W_original]

取 [0, 0]
[H_original, W_original]
```

最终修复：

```python
score_map = torch.abs(
    x - x_hat
).mean(
    dim=1,
    keepdim=True,
)

score_map = F.interpolate(
    score_map,
    size=original_size,
    mode="bilinear",
    align_corners=False,
)

score_map = (
    score_map[0, 0]
    .detach()
    .cpu()
    .numpy()
)
```

---

# 12. Protocol Repair 后结果

一次修复后的训练结果：

```text
Mean Pixel AUROC:
0.798575

Mean Pixel AP:
0.195532

Global Pixel AUROC:
0.791979

Global Pixel AP:
0.191269
```

但此时发现另一个实验方法问题：

```text
为了修 Evaluation
又重新训练了一次模型
```

由于 CUDA 存在一定 nondeterminism：

```text
新模型 ≠ 原模型
```

因此：

> 不能把前后指标差异全部归因于 Evaluation Protocol 修改。

---

# 13. Checkpoint 工程修复

为避免：

```text
每修一个 evaluation bug
就重新训练模型
```

增加：

```text
Checkpoint Save
+
Eval-only Mode
```

Checkpoint：

```text
results/exp025/autoencoder_bottle_minimal/
autoencoder_seed42.pt
```

模式：

```python
EVAL_ONLY = False
```

表示：

```text
Train
→ Save Checkpoint
→ Evaluate
```

模式：

```python
EVAL_ONLY = True
```

表示：

```text
Load Checkpoint
→ Evaluate
```

---

# 14. Canonical Autoencoder Baseline

最终固定 EXP025 Autoencoder：

```text
Seed          = 42
Input         = 224
Batch Size    = 8
Epochs        = 20
LR            = 1e-3
Loss          = L1
```

最终结果：

```text
Mean Pixel AUROC   = 0.80395455
Mean Pixel AP      = 0.19621835

Global Pixel AUROC = 0.79718079
Global Pixel AP    = 0.19078557
```

记为：

```text
EXP025 AE Canonical Baseline
```

---

# 15. Checkpoint Reproducibility 验证

训练后第一次 Evaluation：

```text
Mean Pixel AUROC:
0.8039545476

Mean Pixel AP:
0.1962183587

Global Pixel AUROC:
0.7971807939

Global Pixel AP:
0.1907855693
```

重新加载同一 Checkpoint：

```text
Mean Pixel AUROC:
0.8039545461

Mean Pixel AP:
0.1962183518

Global Pixel AUROC:
0.7971807932

Global Pixel AP:
0.1907855695
```

差异约：

```text
1e-9
```

因此：

```text
Checkpoint Reproducibility
PASS
```

这意味着：

```text
固定模型
+
固定数据
+
固定 Evaluation
→
可复现指标
```

---

# 16. DINOv3 与 Autoencoder 同协议比较

DINOv3 224：

```text
Mean Pixel AUROC:
0.995295

Mean Pixel AP:
0.854635
```

Autoencoder 224：

```text
Mean Pixel AUROC:
0.803955

Mean Pixel AP:
0.196218
```

差值：

```text
Pixel AUROC:

+0.191340
```

```text
Pixel AP:

+0.658416
```

Pair-wise：

```text
22 / 22 images

DINOv3
>
Autoencoder
```

因此初始 Prediction 1：

```text
DINOv3 Patch-NN
优于基础 Reconstruction AE
```

结果：

```text
SUPPORTED
```

---

# 17. 方法对比结论边界

本实验支持：

> 在当前 MVTec Bottle / broken_small / 224 输入条件下，DINOv3 Patch-NN 对 small defect 的 Pixel-level localization 明显优于当前基础 Convolutional Autoencoder。

但不能写成：

> Self-Supervised Learning 导致性能提升。

因为两种方法同时存在以下差异：

```text
Architecture
Pretraining
Representation
Training Objective
Memory Matching
Anomaly Score
```

所以这是：

```text
Method-Class Benchmark
```

而不是：

```text
Causal SSL Ablation
```

---

# 18. Autoencoder Failure Case

Canonical AE 最差 Pixel AP 样本之一：

```text
015.png
008.png
005.png
013.png
000.png
```

典型 AP：

```text
015:
0.05557

008:
0.06930

005:
0.07740

013:
0.08738

000:
0.07286
```

---

# 19. Failure Diagnosis 预测

在进行可视化前，先提出预测。

## Prediction A

高分 False Positive 可能集中于：

```text
Bottle Contour
Ring Edge
Reflection
High-Contrast Structure
```

原因：

```text
这些区域本身难以重建
```

---

## Prediction B

真实缺陷：

```text
GT+ Mean
>
GT- Mean
```

但：

```text
High-score Negative Tail
```

会大量进入异常 Ranking 前部，从而：

```text
AP 很低
```

---

## Prediction C

部分失败样本中：

```text
最大 Reconstruction Error
```

可能甚至不在 GT 内，而是在正常结构区域。

---

# 20. Failure Visualization

生成：

```text
Original
Reconstruction
|x - x_hat|
GT
GT + Anomaly Overlay
```

主要观察：

```text
原图：
瓶口具有大量细环纹、边缘、反光、高频细节

Reconstruction：
明显更加平滑
```

结果：

```text
Anomaly Map
不仅在真实缺陷处高分

同时在大量正常瓶口结构处
也产生高分
```

---

# 21. Failure Diagnosis 定量结果

---

## 21.1 015.png

```text
Pixel AUROC:
0.75050

Pixel AP:
0.05557

GT+ Mean:
0.05288

GT- Mean:
0.02337
```

虽然：

```text
GT+ > GT-
```

但：

```text
198,630 个 Normal Pixel
>
GT Positive Median
```

Precision@GT-size：

```text
0.0755
```

Top-1% high-score Negative：

```text
>100 px from GT:
84.3%
```

说明：

```text
主要 FP
并不是单纯贴着 GT Boundary
```

---

## 21.2 008.png

```text
Pixel AUROC:
0.79314

Pixel AP:
0.06930
```

最大 anomaly score：

```text
Max inside GT:
False
```

Top 0.1% FP：

```text
>100 px:
100%
```

Top 1% FP：

```text
>100 px:
92.9%
```

这是最典型的：

```text
正常结构 Reconstruction Error
>
真实异常
```

案例。

---

## 21.3 005.png

```text
Pixel AUROC:
0.76748

Pixel AP:
0.07740
```

Top 1% FP：

```text
>100 px:
74.8%
```

---

## 21.4 013.png

```text
Pixel AUROC:
0.77086

Pixel AP:
0.08738
```

Top 1% FP：

```text
>100 px:
84.2%
```

---

## 21.5 000.png

```text
Pixel AUROC:
0.81112

Pixel AP:
0.07286
```

Top 1% FP：

```text
>100 px:
83.3%
```

---

# 22. Failure Diagnosis 结论

Prediction A：

```text
正常 contour / reflection /
high-frequency structure
会产生高分 FP
```

结果：

```text
SUPPORTED
```

Prediction B：

```text
High-score Negative Tail
导致低 AP
```

结果：

```text
STRONGLY SUPPORTED
```

Prediction C：

```text
Maximum Error
可能不在 GT
```

结果：

```text
PARTIALLY SUPPORTED
```

例如：

```text
008.png:
Max inside GT = False
```

其余部分样本：

```text
Max inside GT = True
```

但单个最大值并不能代表整体 Ranking 质量。

---

# 23. Primary Failure Mechanism

当前证据支持：

```text
Primary Failure：

Normal-Structure Reconstruction Mismatch
```

即：

```text
Autoencoder
无法精确重建正常高频结构

→ 正常区域 Reconstruction Error 高

→ High-score False Positive

→ 污染异常 Ranking

→ Pixel AP 大幅下降
```

当前证据不足以支持：

```text
异常完全被 Autoencoder 重建掉
```

作为主要原因。

原因：

```text
GT+ Mean
仍然普遍高于 GT- Mean
```

并且部分样本：

```text
最大 anomaly score
仍位于 GT 内
```

---

# 24. Active Modification #1：Autoencoder Capacity Ablation

为了验证：

```text
是否因为模型容量不足
导致正常结构重建不好
```

主动修改：

```text
Channel Capacity
```

其余变量保持一致。

---

# 25. Capacity Ablation 预测

## Small

```text
16
→ 32
→ 64
→ 128
```

预测：

```text
正常重建更差
→ Normal Structural FP 更多
→ Pixel AP 更低
```

---

## Base

```text
32
→ 64
→ 128
→ 256
```

作为 Reference。

---

## Large

```text
64
→ 128
→ 256
→ 512
```

预测：

```text
正常重建可能改善
→ Normal FP 下降
→ AP 可能提高
```

但：

```text
如果异常也被重建得更好
→ GT+ Error 下降
→ Anomaly Contrast 可能停止提高
```

因此预测：

```text
性能不一定单调增长
```

---

# 26. Capacity Ablation 结果

| Model | Channels | Parameters | Normal Test L1 | Mean Pixel AUROC | Mean Pixel AP |
|---|---|---:|---:|---:|---:|
| Small | 16-32-64-128 | 270,355 | 0.03028 | 0.75015 | 0.16262 |
| Base | 32-64-128-256 | 1,078,307 | 0.02506 | 0.79955 | 0.19252 |
| Large | 64-128-256-512 | 4,307,011 | 0.02228 | 0.80153 | 0.19280 |

---

# 27. Small → Base 分析

Normal Test L1：

```text
0.03028
→
0.02506
```

下降约：

```text
17.2%
```

Mean Pixel AP：

```text
0.16262
→
0.19252
```

提升约：

```text
18.4%
```

Pair-wise：

```text
AP:
21 / 22 improved

AUROC:
22 / 22 improved
```

同时 Failure subset 中：

```text
Negative > Positive Median
明显减少
```

因此：

> Small AE 的确存在明显的 under-capacity 问题。

结论：

```text
Under-Capacity
→ Normal Reconstruction Mismatch
→ Structural FP
→ Lower AP
```

得到支持。

---

# 28. Base → Large 分析

Normal Test L1：

```text
0.02506
→
0.02228
```

继续下降约：

```text
11.1%
```

但 Mean Pixel AP：

```text
0.19252
→
0.19280
```

几乎不变。

GT- Mean：

```text
0.02382
→
0.02101
```

正常区域重建更好。

但 GT+ Mean：

```text
0.08831
→
0.08286
```

异常区域也被重建得更好。

因此：

```text
模型容量继续增加

→ Normal Error ↓
同时
→ Anomaly Error ↓
```

最终：

```text
Anomaly Ranking 改善进入平台
```

---

# 29. Cause Validation #1

假设：

```text
Autoencoder Capacity 不足
是正常结构 False Positive 的原因之一
```

实验：

```text
Small
→ Base
→ Large
```

结果：

```text
Small → Base:
明显改善

Base → Large:
进入 Saturation
```

结论：

> Under-capacity 是当前 Autoencoder Failure 的一个重要原因，但并不是唯一原因。

状态：

```text
CAUSE VALIDATION #1
PASS
```

---

# 30. Active Analysis：High-Gradient FP

Failure Visualization 显示：

```text
大量高分 FP
位于瓶口边缘、环纹和反光结构
```

因此提出假设：

> Autoencoder 的 high-score normal pixels 是否显著集中于 high-gradient normal regions？

---

# 31. Edge FP Analysis

对每张原图计算：

```text
Sobel Gradient Magnitude
```

然后只分析：

```text
GT-negative pixels
```

比较：

```text
A:
所有 Normal Pixel Gradient

B:
Anomaly Score Top-1% Normal Pixel Gradient
```

另外定义：

```text
Normal Pixel 中
Gradient Top-10%
```

检查：

```text
Top-1% FP 中
有多少同时属于
Top-10% High-Gradient Region
```

---

# 32. Edge FP Analysis 预测

如果：

```text
高梯度正常结构
与 Autoencoder False Positive 强相关
```

那么应该出现：

```text
Top-1% FP Gradient
>
Normal Gradient
```

且：

```text
Top-1% FP
落入 Top-10% Gradient 的比例
>>
10%
```

---

# 33. Edge FP Analysis 结果

22 张图像：

```text
Mean Gradient Ratio:

2.8939×
```

即：

```text
Top-1% FP 的平均 Gradient

≈

普通 Normal Pixel 的 2.89 倍
```

Median：

```text
2.9285×
```

---

Top-1% FP 中：

```text
33.2%
```

落入：

```text
Normal Pixel Top-10%
High-Gradient Region
```

随机基线：

```text
10%
```

因此 Enrichment：

```text
3.3204×
```

并且：

```text
22 / 22 images

Gradient Ratio > 1
Enrichment > 1
```

---

# 34. Edge FP Analysis 结论

实验支持：

> 当前 Autoencoder 的 high-score normal pixels 显著富集在 high-gradient normal structures 中。

例如：

```text
瓶口边缘
环纹
高光
反光
高对比纹理
```

但是：

```text
这是 Association Evidence
```

不能仅凭此实验写成：

```text
High Gradient
严格因果导致
False Positive
```

因为：

```text
本实验没有主动修改 Gradient
```

因此正式结论：

```text
High-gradient normal structures
are strongly associated with
Autoencoder false positives.
```

状态：

```text
SUPPORTED
```

---

# 35. Cause Validation #2：DINOv3 Spatial Resolution

EXP024 中已经进行控制变量实验：

```text
DINOv3

Input:
224
→
448
```

保持：

```text
模型
异常算法
数据集
测试图像
主要计算流程
```

一致。

主要改变：

```text
Spatial Granularity
```

224：

```text
14 × 14 Patch Grid
```

448：

```text
28 × 28 Patch Grid
```

---

# 36. Resolution Ablation 结果

224：

```text
Mean Pixel AUROC:
0.995295

Mean Pixel AP:
0.854635
```

448：

```text
Mean Pixel AUROC:
0.998341

Mean Pixel AP:
0.951768
```

Mean AP Delta：

```text
+0.097133
```

Pair-wise：

```text
Wins / Ties / Losses

21 / 0 / 1
```

Bootstrap 95% CI：

```text
[+0.067179, +0.140345]
```

因此：

```text
CAUSE VALIDATION #2
PASS
```

结论：

> Spatial Granularity 是当前 small-defect localization 的重要影响因素。

---

# 37. 特殊 Failure：013.png

DINOv3 013：

```text
224 → 448
```

出现：

```text
AUROC:
提高

AP:
下降
```

AP：

```text
0.8774
→
0.8662
```

说明：

```text
整体 Positive / Negative Separation
可以改善

但 Top Ranking 中的 FP
仍可能增加
```

空间诊断发现：

```text
High-score Negative Tail
主要贴近 GT Boundary
```

因此：

```text
Boundary / Annotation Sensitivity
```

是该样本的重要 Failure。

---

# 38. AE 与 DINOv3 Failure Mechanism 对比

DINOv3 013：

```text
高分 FP
主要位于 GT Boundary 附近
```

Autoencoder 013：

```text
Top-1% FP 中
84.2%
距离 GT >100 px
```

因此：

```text
两种方法不是
“同一种 Failure 的程度不同”
```

而是可能存在：

```text
不同 Failure Mechanism
```

DINOv3：

```text
Boundary / Patch Granularity
```

Autoencoder：

```text
Normal Structural Reconstruction Mismatch
```

---

# 39. Top-5 Failure Modes

## Failure Mode 1

```text
AE Normal Structural False Positive
```

表现：

```text
正常瓶口边缘、环纹、反光
产生高 Reconstruction Error
```

证据：

```text
Top-1% FP Gradient
= 2.89× Normal Gradient

High-gradient Enrichment
= 3.32×
```

---

## Failure Mode 2

```text
AE Under-Capacity Reconstruction Mismatch
```

证据：

```text
Small → Base

Normal L1 ↓
AP ↑
AUROC ↑
```

---

## Failure Mode 3

```text
AE Capacity Saturation /
Anomaly Co-Reconstruction
```

证据：

```text
Base → Large

Normal L1
继续下降

但 AP
几乎不变

同时 GT+ Error
也下降
```

---

## Failure Mode 4

```text
Limited Spatial Granularity
```

证据：

```text
DINOv3

224 → 448

Mean AP:
+0.09713

21 / 22 improved
```

---

## Failure Mode 5

```text
Boundary / Annotation Sensitivity
```

典型：

```text
DINOv3 013.png
```

现象：

```text
AUROC ↑
AP ↓
```

原因分析：

```text
Top-ranked FP
聚集在 GT Boundary
```

---

# 40. AUROC 与 AP 的进一步理解

Autoencoder 典型现象：

```text
Pixel AUROC ≈ 0.8

Pixel AP ≈ 0.2
```

说明：

```text
模型具有一定整体排序能力
```

但：

```text
Top Ranking
被大量 False Positive 污染
```

对于 small defect：

```text
Positive Pixel
远少于 Negative Pixel
```

因此 AP 对：

```text
High-score FP
```

非常敏感。

结论：

> AUROC 尚可，并不代表 anomaly localization 足够好。

---

# 41. Mean Metric 与 Global Metric

Mean Per-Image：

```text
每张图先独立计算指标
→ 22 张再平均
```

因此：

```text
每张图权重相同
```

Global：

```text
把 22 张图 Pixel 全部合并
→ 再计算 AUROC/AP
```

因此：

```text
每个 Pixel 权重相同
```

两者回答的问题不同。

不能：

```text
静默替换
```

---

# 42. 一个重要实验方法教训

第一次 Protocol Repair 时：

```text
只想修改 Evaluation
```

但由于：

```text
没有保存 Checkpoint
```

不得不重新训练。

导致：

```text
Evaluation Protocol
+
Model Weights

同时变化
```

从而破坏：

```text
单变量因果解释
```

因此以后必须：

```text
Training
与
Evaluation
解耦
```

基本规范：

```text
训练完成
→ 保存 Checkpoint

Evaluation 修改
→ Load Same Checkpoint
→ Eval-only
```

---

# 43. Reproducibility Lesson

```text
seed = 42
```

并不自动意味着：

```text
GPU 训练完全 deterministic
```

因为：

```text
CUDA
cuDNN
某些算子
```

可能存在 nondeterminism。

因此正式实验应同时考虑：

```text
Random Seed
Checkpoint
Data Order
Evaluation Protocol
Library Version
Hardware
Deterministic Settings
```

---

# 44. 工业方法选择结论

当前测试条件：

```text
MVTec Bottle
broken_small
small localized defect
```

当前结果支持：

```text
DINOv3 Patch Representation
+
Local Nearest-Neighbor Matching
```

比：

```text
Basic Convolutional Autoencoder
+
Pixel Reconstruction Error
```

更适合作为 small defect localization 的候选方案。

---

# 45. 但工业部署不能只看 Accuracy

实际生产选择还需要考虑：

```text
Latency
GPU / CPU Requirement
VRAM
Feature Memory Size
Throughput
Threshold Stability
Reference Data Maintenance
Defect Distribution
Annotation Quality
Deployment Complexity
Maintainability
```

DINOv3 优点：

```text
Representation 强
Pixel Localization 强
无需异常训练数据
```

代价：

```text
模型更重
Feature Memory 有存储成本
Nearest Neighbor 有推理成本
Resolution 提高后成本继续增加
```

Autoencoder 优点：

```text
结构简单
推理逻辑简单
容易部署
```

问题：

```text
Normal Structural FP
Reconstruction Assumption 不稳定
Capacity Trade-off
```

---

# 46. 当前实验脚本

## Canonical Autoencoder

```text
src/experiments/
exp025_autoencoder_bottle_minimal.py
```

---

## Failure Diagnosis

```text
src/experiments/
exp025_autoencoder_failure_diagnosis.py
```

---

## Capacity Ablation

```text
src/experiments/
exp025_autoencoder_capacity_ablation.py
```

---

## Edge FP Analysis

```text
src/experiments/
exp025_autoencoder_edge_fp_analysis.py
```

---

# 47. 当前结果目录

```text
results/exp025/
```

主要目录：

```text
autoencoder_bottle_minimal/
autoencoder_failure_diagnosis/
autoencoder_capacity_ablation/
autoencoder_edge_fp_analysis/
```

Canonical Checkpoint：

```text
results/exp025/
autoencoder_bottle_minimal/
autoencoder_seed42.pt
```

---

# 48. 最终核心结果汇总

## Method Comparison

| Method | Mean Pixel AUROC | Mean Pixel AP |
|---|---:|---:|
| DINOv3 224 | 0.995295 | 0.854635 |
| Autoencoder 224 | 0.803955 | 0.196218 |

---

## Autoencoder Capacity Ablation

| Model | Normal Test L1 | Mean Pixel AP |
|---|---:|---:|
| Small | 0.03028 | 0.16262 |
| Base | 0.02506 | 0.19252 |
| Large | 0.02228 | 0.19280 |

---

## DINOv3 Resolution Ablation

| Input | Mean Pixel AUROC | Mean Pixel AP |
|---|---:|---:|
| 224 | 0.995295 | 0.854635 |
| 448 | 0.998341 | 0.951768 |

---

## Edge FP Analysis

```text
Mean Gradient Ratio:
2.8939×

Mean High-gradient Enrichment:
3.3204×

22 / 22 images:
enrichment > 1
```

---

# 49. 关键知识总结

## 49.1 Reconstruction Error 的前提

重建式 anomaly detection 隐含假设：

```text
Normal
比
Anomaly
更容易重建
```

如果：

```text
正常高频结构也难重建
```

则会：

```text
Normal FP ↑
```

如果：

```text
模型容量过强
异常也被很好重建
```

则会：

```text
Anomaly Contrast ↓
```

---

## 49.2 更低的训练 Loss 不一定更好

实验已经直接证明：

```text
Base → Large

Normal Reconstruction
继续改善

但 Pixel AP
几乎不再改善
```

因此：

```text
Training Objective
≠
Final Evaluation Objective
```

---

## 49.3 Spatial Resolution 是算法参数

对于 Patch-based localization：

```text
Input Resolution
```

不是简单的数据预处理参数。

它会决定：

```text
Patch Grid
Spatial Granularity
Feature Memory Size
Compute Cost
Boundary Behavior
```

因此属于核心实验变量。

---

## 49.4 Failure Analysis 不能只看 Heatmap

正确过程：

```text
可视化
→ 提出假设
→ 定义 Quantitative Metric
→ 主动修改变量
→ 验证预测
```

例如本实验：

```text
Heatmap 看见边缘 FP
↓
提出 High-gradient 假设
↓
Sobel Gradient Quantification
↓
得到 2.89× / 3.32×
```

---

# 50. 当前掌握情况

目前已经能够解释：

```text
Autoencoder anomaly detection 原理
DINOv3 Patch-NN 原理
AUROC 与 AP 差异
Mean 与 Global Metric 差异
Checkpoint reproducibility
Evaluation geometry
Capacity trade-off
Spatial granularity
High-score negative tail
Boundary sensitivity
Failure taxonomy
Cause validation
```

目前已经能够预测：

```text
模型容量过小会怎样
模型容量继续增加会怎样
分辨率提高会怎样
High-score FP 对 AP 会怎样
Boundary FP 对 AUROC/AP 会怎样
```

目前已经能够主动修改：

```text
AE Channel Capacity
DINOv3 Input Resolution
Evaluation Protocol
Checkpoint / Eval-only Workflow
```

目前已经能够排错：

```text
Tensor Shape
Undefined score_map
GT / score geometry mismatch
Evaluation protocol mismatch
Checkpoint reproducibility
实验变量混淆
```

---

# 51. 仍需进一步稳定的能力

以下内容还需要通过 BOSS 答辩进一步验证：

```text
1. 能否脱离笔记完整解释两类方法

2. 能否解释为什么
   Lower Reconstruction Loss
   不等于 Better Anomaly Detection

3. 能否解释 Small → Base
   和 Base → Large
   为什么表现不同

4. 能否解释
   AUROC ↑
   但 AP ↓

5. 能否根据工业要求
   做 Accuracy / Latency / Memory Trade-off

6. 能否设计下一轮实验
   而不是直接乱调参数
```

---

# 52. SOP 主动修改记录

本 Learning Unit 已完成多次主动修改：

```text
Modification 1:
Evaluation Protocol
224 GT
→ Original-resolution GT

Modification 2:
Checkpoint / Eval-only
Train+Eval Coupling
→ Train/Eval Decoupling

Modification 3:
Autoencoder Capacity
Small / Base / Large

Modification 4:
DINOv3 Resolution
224 / 448
```

满足：

```text
主动修改 >= 3
```

---

# 53. 原因验证状态

```text
Cause #1:
Autoencoder Under-Capacity
→ VALIDATED

Cause #2:
DINOv3 Spatial Granularity
→ VALIDATED

Additional Evidence:
High-gradient Normal Structure
与 AE False Positive 强相关
→ SUPPORTED
```

---

# 54. Level 1 BOSS 最终验收状态

```text
两种不同方法类别                  PASS
同数据集对比                      PASS
统一 Evaluation Protocol         PASS
Baseline                          PASS
主动修改 >= 3                     PASS
至少 3 个 Ablation                PASS
Top-5 Failure Modes              PASS
验证 >= 2 个 Failure 原因         PASS
Checkpoint Reproducibility       PASS
Failure Analysis                 PASS
可复现实验命令                    PASS
README                            PASS

方法原理与核心假设答辩             PASS
Capacity / Loss / Ranking 答辩    PASS
AUROC / AP / Boundary FP 答辩     PASS
实现 / Debug / Protocol 答辩      PASS
工业迁移 / Trade-off 答辩         PASS

10 分钟技术答辩                    PASS
最终知识验收                       PASS
核心文件检查                       PASS
Python 语法检查                    PASS
Selective Git Staging            PASS

Git Commit                        PENDING
GitHub Push                       PENDING

---
# 55. 当前 Level 状态

```text
Level 1
CONTINUE