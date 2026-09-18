# EXP017 - Global vs Patch Bottle Evaluation

## 1. Goal

本实验的目标是：

在完全相同的 MVTec AD Bottle Test Set 上，直接比较：

```text
Global Feature Baseline
vs
Patch Coreset100 Baseline
```

重点回答：

1. Global Representation 与 Patch Representation 在 image-level anomaly detection 上整体表现如何？
2. 两种方法在 `broken_large`、`broken_small`、`contamination` 三类 defect 上是否具有不同偏好？
3. Patch 保留局部空间信息，是否一定带来更高 image-level AUROC？
4. Global Pooling 虽然损失空间信息，是否仍可能保留足够的 anomaly signal？
5. 两种方法的 Failure Candidates 是否重合？
6. 当前结果可以支持哪些结论，又有哪些因果关系不能直接下结论？

---

## 2. Environment

```text
Project:
Project_001_Baseline_Experiment

Conda Environment:
ai_base

Framework:
PyTorch

Evaluation:
scikit-learn

Backbone:
ResNet18

Pretrained Weights:
ImageNet

Input Size:
224 × 224

Device:
CPU

Dataset:
MVTec AD

Category:
bottle
```

---

## 3. Test Set

本实验使用与 EXP016 完全相同的：

```text
83 Bottle Test Images
```

组成：

```text
good           20
broken_large   20
broken_small   22
contamination  21

Total          83
```

Label：

```text
good     = 0
defect   = 1
```

Patch Score 直接复用 EXP016：

```text
Coreset100
```

因此 Patch Test Set 与 EXP016 完全一致。

---

## 4. Compared Methods

### 4.1 Global Baseline

Global Baseline：

```text
Image
↓
ResNet18
↓
layer4
↓
Global Average Pooling
↓
512-D Global Feature
↓
Normal Center
↓
L2 Distance
↓
Image Anomaly Score
```

---

### 4.2 Patch Baseline

Patch Baseline：

```text
Image
↓
ResNet18 layer2
↓
28 × 28 Spatial Feature Map
↓
784 × 128-D Patch Features
↓
Coreset100
↓
Nearest Neighbor Distance
↓
Max Patch Score
↓
Image Anomaly Score
```

---

## 5. Important Experimental Limitation

EXP017 并不是严格的：

```text
Global Pooling
vs
Patch Representation
```

单变量 Ablation。

因为两种方法同时存在以下差异：

| Global Baseline | Patch Baseline |
|---|---|
| layer4 | layer2 |
| 512-D | 128-D patch |
| global pooled feature | spatial patch feature |
| normal center | memory bank |
| center distance | nearest-neighbor distance |
| whole-image representation | max patch aggregation |

因此 EXP017 可以回答：

> 当前两套完整 baseline 在 Bottle Dataset 上分别表现如何。

但不能严格证明：

```text
Global 更好
因为 Global Pooling 更好
```

或者：

```text
Patch 在某类更好
完全因为保留空间信息
```

这些原因需要后续 Ablation 才能进一步验证。

---

## 6. Global Feature Extraction

标准 ResNet18 forward：

```text
Image
↓
conv1
↓
layer1
↓
layer2
↓
layer3
↓
layer4
↓
avgpool
↓
flatten
↓
fc
↓
1000 ImageNet logits
```

本实验：

```python
model.fc = torch.nn.Identity()
```

因此最后分类层被移除。

模型直接输出：

```text
[B,512]
```

即：

```text
Global Image Feature
```

实际结果：

```text
Train normal images:
209

Train global features:
[209,512]

Test global features:
[83,512]
```

---

## 7. Global Normal Center

209 个 Normal Train Feature：

\[
f_1,f_2,\ldots,f_{209}
\]

每个：

\[
f_i\in\mathbb{R}^{512}
\]

定义 Normal Center：

\[
c
=
\frac{1}{209}
\sum_{i=1}^{209}f_i
\]

实际 Shape：

```text
[512]
```

可以理解为：

> Normal Bottle 在 Global Feature Space 中的平均位置。

---

## 8. Global Anomaly Score

对于 Test Image：

\[
f(x)\in\mathbb{R}^{512}
\]

Global anomaly score：

\[
S_G(x)
=
\|f(x)-c\|_2
\]

如果 Test Feature 接近 Normal Center：

```text
score low
```

如果 Test Feature 明显偏离正常中心：

```text
score high
```

实际 Test Scores Shape：

```text
[83]
```

即：

```text
1 image
→
1 global anomaly score
```

---

## 9. Patch Anomaly Score

Patch Baseline 沿用 EXP016。

每张图片：

```text
28 × 28
=
784 patches
```

Patch score：

\[
s(q_i)
=
\min_{m\in M}
\|q_i-m\|_2
\]

其中：

```text
M = Coreset100
```

Image score：

\[
S_P(x)
=
\max_i s(q_i)
\]

因此：

```text
Global:
整张图整体 representation 偏离多少？

Patch:
是否存在某个局部区域无法被 normal patches 解释？
```

---

## 10. Prediction

### Prediction 1

之前 Global Bottle Baseline 已得到约：

```text
AUROC ≈ 0.973
```

因此预计本实验 Global AUROC 应接近该结果。

如果差异很大，需要检查：

```text
preprocess
feature extraction
test ordering
score implementation
```

---

### Prediction 2

EXP016 Patch Coreset100：

```text
Overall AUROC:
0.947619
```

因此本实验不预测 Patch 在 Overall AUROC 上一定优于 Global。

---

### Prediction 3

Patch 保留局部 Spatial Response。

因此：

```text
broken_small
```

可能是 Patch 相对更有竞争力的 defect type。

但是否真正优于 Global 必须通过实验验证。

---

### Prediction 4

EXP016 表明：

```text
contamination
```

是 Patch Baseline 最困难的 defect type。

因此需要检查：

```text
Global 是否也在 contamination 上困难？
```

如果 Global contamination 明显更好，说明不同 representation / scoring pipeline 对 defect type 存在明显不同的敏感性。

---

### Prediction 5

即使 Patch Overall AUROC 不高于 Global，Patch 仍提供：

```text
spatial anomaly map
localization
top-k patch response
```

而 Global Baseline 本身不直接提供这些空间定位能力。

因此：

```text
Image-level classification
```

与：

```text
Anomaly localization
```

必须分开评价。

---

## 11. Global Baseline Result

### Overall

```text
Overall AUROC:
0.974603

Average Precision:
0.990410
```

### GOOD

```text
Mean:
4.027867

P95:
4.909111

Max:
7.677730
```

### ALL DEFECTS

```text
Min:
4.154656

Mean:
9.165263

Max:
15.003535
```

### Separation Margin

\[
4.154656-7.677730
=
-3.523074
\]

结果：

```text
Margin < 0
```

说明存在 Score Distribution Overlap。

---

## 12. Patch Coreset100 Result

### Overall

```text
Overall AUROC:
0.947619

Average Precision:
0.978718
```

### GOOD

```text
Mean:
3.777570

P95:
4.361700

Max:
4.635723
```

### ALL DEFECTS

```text
Min:
3.836864

Mean:
4.641121

Max:
6.448766
```

### Separation Margin

```text
-0.798858
```

---

## 13. Important Rule - Do Not Compare Raw Score Magnitude

Global：

```text
512-D layer4 global feature
distance to normal center
```

Patch：

```text
128-D layer2 patch feature
distance to nearest memory patch
```

因此：

```text
Global Score = 7
Patch Score  = 4
```

不能解释为：

```text
Global 更异常
```

两种 Raw Score 不在同一个 Feature Space，也不在同一个 Scoring System 中。

可比较的指标包括：

```text
AUROC
AP
Ranking
Failure Pattern
Per-defect Performance
```

不能直接比较：

```text
Raw Score Magnitude
```

---

## 14. Overall Comparison

| Metric | Global | Patch Coreset100 |
|---|---:|---:|
| AUROC | 0.974603 | 0.947619 |
| Average Precision | 0.990410 | 0.978718 |

AUROC 差：

\[
0.974603-0.947619
=
0.026984
\]

即：

```text
Global
约高 2.7 AUROC percentage points
```

---

## 15. Pairwise Interpretation of AUROC

Test Set：

```text
20 good
63 defect
```

总 Good-Defect Pair：

\[
20\times63
=
1260
\]

Global：

\[
0.974603\times1260
\approx1228
\]

约：

```text
1228 / 1260
```

pair ranking 正确。

Patch：

\[
0.947619\times1260
=
1194
\]

约：

```text
1194 / 1260
```

pair ranking 正确。

两者相差约：

```text
34 pair rankings
```

因此 Global 的 Overall Ranking Advantage 并非只来自 1～2 个偶然 pair。

---

## 16. Per-Defect Result - Global

### broken_large

```text
AUROC = 0.990000
AP    = 0.989710
```

### broken_small

```text
AUROC = 0.956818
AP    = 0.947357
```

### contamination

```text
AUROC = 0.978571
AP    = 0.978470
```

---

## 17. Per-Defect Result - Patch

### broken_large

```text
AUROC = 0.952500
AP    = 0.942552
```

### broken_small

```text
AUROC = 0.959091
AP    = 0.954006
```

### contamination

```text
AUROC = 0.930952
AP    = 0.926809
```

---

## 18. Per-Defect Comparison

| Defect Type | Global AUROC | Patch AUROC |
|---|---:|---:|
| broken_large | 0.990000 | 0.952500 |
| broken_small | 0.956818 | 0.959091 |
| contamination | 0.978571 | 0.930952 |

---

## 19. Broken Large Analysis

Global：

```text
0.990000
```

Patch：

```text
0.952500
```

这里：

```text
20 good
20 broken_large
```

共：

\[
20\times20=400
\]

pair。

Global：

\[
0.99\times400
=
396
\]

正确 pair：

```text
396 / 400
```

Patch：

\[
0.9525\times400
=
381
\]

正确 pair：

```text
381 / 400
```

相差：

```text
15 pairs
```

说明 Global 对 `broken_large` 的 image-level ranking 明显更强。

可能原因之一是：

```text
broken_large
```

影响较大空间范围，因此能够明显改变 Whole-image Representation。

即使经过 Global Average Pooling：

```text
异常信号仍然可能保留
```

但该解释目前属于合理 hypothesis，而非严格单变量证明。

---

## 20. Broken Small Analysis

Global：

```text
0.956818
```

Patch：

```text
0.959091
```

这里：

```text
20 good
22 broken_small
```

共：

\[
20\times22=440
\]

pair。

Global：

\[
0.956818\times440
\approx421
\]

Patch：

\[
0.959091\times440
\approx422
\]

因此只相差约：

```text
1 pair
```

所以不能写：

```text
Patch 明显优于 Global
```

更准确：

> Global 与 Patch 在 broken_small 的 image-level ranking performance 上基本相当，Patch 数值略高，但仅相差约一个 pair ranking。

---

## 21. Contamination Analysis

Global：

```text
0.978571
```

Patch：

```text
0.930952
```

这里：

```text
20 good
21 contamination
```

共：

\[
20\times21=420
\]

pair。

Global：

\[
0.978571\times420
\approx411
\]

Patch：

\[
0.930952\times420
\approx391
\]

相差约：

```text
20 pair rankings
```

因此 Global 在 contamination 上明显更强。

这非常重要，因为 EXP016 已经发现：

```text
contamination
```

是当前 Patch Baseline 的主要 weakness。

而 Global Baseline 对 contamination：

```text
AUROC = 0.978571
```

说明：

> contamination 并不是对所有 ResNet18-based representations 都同样困难。

当前困难更可能与：

```text
Patch pipeline
```

的某些组成部分有关，例如：

```text
layer2 feature
nearest-neighbor matching
Coreset representation
max aggregation
local texture similarity
```

具体原因仍需进一步 Ablation。

---

## 22. Why Global Can Outperform Patch

Global Average Pooling 会丢失 Spatial Information。

但：

```text
Loss of Spatial Information
```

不等于：

```text
Loss of All Anomaly Information
```

如果异常：

```text
影响范围较大
或
引起强 feature response
或
改变整体纹理 / 结构统计
```

那么经过 Global Pooling 后：

```text
Global Representation
```

仍然可能明显偏离 Normal Center。

因此：

```text
Global
```

的异常假设更接近：

> 整张图片的 representation 是否发生整体 shift？

而 Patch：

> 是否存在某个局部区域无法被 normal local features 解释？

这是两种不同的 anomaly assumptions。

---

## 23. Global Failure Candidates

### Top-5 Highest-Scoring GOOD

```text
1. good/006.png    7.677730
2. good/008.png    4.763395
3. good/018.png    4.695288
4. good/015.png    4.483432
5. good/019.png    4.277589
```

---

### Top-5 Lowest-Scoring DEFECT

```text
1. broken_small/007.png     4.154656
2. contamination/003.png    4.680594
3. broken_small/000.png     4.896986
4. contamination/019.png    4.898993
5. broken_small/021.png     5.081191
```

---

## 24. Patch Failure Candidates

### Top-5 Highest-Scoring GOOD

```text
1. good/010.png    4.635723
2. good/006.png    4.347278
3. good/001.png    4.173130
4. good/016.png    3.885491
5. good/009.png    3.880455
```

---

### Top-5 Lowest-Scoring DEFECT

```text
1. contamination/004.png    3.836864
2. contamination/003.png    3.926649
3. contamination/012.png    3.971097
4. contamination/001.png    3.985341
5. contamination/020.png    3.990207
```

---

## 25. Common High-Scoring Normal Candidate

两种方法共同出现：

```text
good/006.png
```

Global：

```text
Rank 1
score = 7.677730
```

Patch：

```text
Rank 2
score = 4.347278
```

因此：

```text
good/006
```

是一个非常重要的：

```text
Common High-score Normal Candidate
```

因为：

```text
Global
+
Patch
```

两种不同 Normal Modeling Strategy 都认为它较异常。

这意味着它可能包含某种：

```text
真实但合法的 Normal Variation
```

但具体原因必须通过图像检查和控制实验验证。

---

## 26. Method-Specific High-Score Normal

Patch 的最高 Normal：

```text
good/010
```

但它没有出现在 Global Top-5。

因此可能存在：

```text
Patch-sensitive Local Variation
```

也就是说：

> 某些局部结构对 Patch NN 来说不常见，但未明显改变 Global Representation。

这是后续 Failure Analysis 的重要对照样本。

---

## 27. Common Low-Scoring Defect

两种方法的 low-score defect 中共同出现：

```text
contamination/003
```

Global：

```text
Rank 2 low-score defect
```

Patch：

```text
Rank 2 low-score defect
```

说明：

```text
contamination/003
```

同时对 Global Center Distance 和 Patch NN Distance 都较难。

因此它是一个重要：

```text
Common Hard Anomaly Candidate
```

后续应优先分析。

---

## 28. Separation Margin Counterexample

Global：

```text
Margin = -3.523074
```

Patch：

```text
Margin = -0.798858
```

如果只看 Margin：

```text
Patch 看起来明显更好
```

但 AUROC：

```text
Global = 0.974603
Patch  = 0.947619
```

结果相反。

原因：

Global 存在一个极端 high-score good：

```text
good/006
score = 7.677730
```

以及 low-score defect：

```text
broken_small/007
score = 4.154656
```

因此：

\[
4.154656-7.677730
=
-3.523074
\]

Margin 被极端样本强烈影响。

而 AUROC 考虑：

```text
全部 1260 个
good-defect pair rankings
```

因此：

> Separation Margin 对 outlier 极其敏感，不能替代 AUROC。

---

## 29. Relationship to EXP016

EXP016 Patch Coreset100：

```text
Overall AUROC:
0.947619
```

EXP017 复用同一 Patch Scores：

```text
Overall AUROC:
0.947619
```

完全一致。

因此：

```text
Patch Evaluation Reuse
```

正确。

---

## 30. Global Baseline Reproduction

之前 Global Bottle Baseline：

```text
AUROC ≈ 0.973015873
```

EXP017：

```text
AUROC = 0.974603
```

差异：

\[
0.974603-0.973016
\approx0.001587
\]

对应 1260 pair 大约：

```text
2 pair rankings
```

因此可以描述：

```text
Prior Global Baseline
approximately reproduced
```

但不能记录为：

```text
Exact reproduction
```

后续正式 Benchmark 整理时应检查：

```text
preprocessing
model construction
feature extraction
test ordering
torchvision version
weights transform
score implementation
```

确认差异来源。

---

## 31. Prediction vs Result

### Prediction 1

预测：

```text
Global Overall AUROC
接近之前约 0.973
```

实际：

```text
0.974603
```

Verdict：

```text
PASS
Approximate Reproduction
```

---

### Prediction 2

预测：

```text
不假设 Patch Overall 优于 Global
```

实际：

```text
Global:
0.974603

Patch:
0.947619
```

Global Overall 更高。

---

### Prediction 3

预测：

```text
Patch 在 broken_small 上可能有竞争力
```

实际：

```text
Global:
0.956818

Patch:
0.959091
```

Patch 数值略高。

但只相差约：

```text
1 pair
```

因此：

```text
基本相当
```

Verdict：

```text
SUPPORTED WITH CAUTION
```

---

### Prediction 4

预测：

```text
需要检查 Global contamination performance
```

实际：

```text
Global contamination:
0.978571

Patch contamination:
0.930952
```

Global 明显更强。

因此 Patch 的 contamination weakness：

```text
不是所有 ResNet18 anomaly pipeline
共有的问题
```

更可能与当前 Patch Pipeline 的某些设计有关。

---

### Prediction 5

预测：

```text
Patch 即使 Overall 不胜，
仍具有 localization capability
```

成立。

Global 当前只输出：

```text
1 image score
```

Patch 同时可以保留：

```text
patch score map
spatial localization
top-k abnormal patches
```

Verdict：

```text
PASS
```

---

## 32. What EXP017 Proves

EXP017 可以支持：

### Conclusion A

在当前 Bottle image-level anomaly detection 上：

```text
Global Baseline Overall Ranking
>
Patch Coreset100 Overall Ranking
```

---

### Conclusion B

不同 defect type 对两套方法的响应不同。

```text
broken_large:
Global 明显更强

broken_small:
两者基本相当

contamination:
Global 明显更强
```

---

### Conclusion C

Global Average Pooling 虽然损失空间信息，但仍能获得非常强的 image-level anomaly detection performance。

---

### Conclusion D

Patch 的价值不能只用 Overall AUROC 判断，因为 Patch 还能提供：

```text
spatial anomaly map
localization
failure-region analysis
```

这些能力 Global Baseline 当前不直接具备。

---

## 33. What EXP017 Does NOT Prove

不能直接证明：

```text
layer4 比 layer2 好
```

不能直接证明：

```text
Global Pooling 比 Patch Feature 好
```

不能直接证明：

```text
Center Model 比 Nearest Neighbor 好
```

不能直接证明：

```text
Coreset 导致 Patch contamination weakness
```

因为当前 Comparison 同时改变多个组件。

这些需要：

```text
Controlled Ablation
```

进一步验证。

---

## 34. Failure Analysis Targets

### Common High-score Normal

```text
good/006
```

优先研究：

```text
为什么 Global 和 Patch 都认为它异常？
```

---

### Patch-specific High-score Normal

```text
good/010
```

优先研究：

```text
为什么 Patch 高分，
而 Global 没进入 Top-5？
```

---

### Common Hard Defect

```text
contamination/003
```

优先研究：

```text
为什么两种方法都低分？
```

---

### Patch-specific Weakness

```text
contamination/004
contamination/012
contamination/001
contamination/020
```

优先研究：

```text
为什么 Patch Score 特别低？
```

---

## 35. Engineering Observation

EXP017 直接复用 EXP016 CSV：

```text
results/exp016/
full_bottle_scores.csv
```

而不是重新运行 Patch Backbone / Memory Search。

这是一个正确的实验工程习惯：

```text
Score Generation
和
Metric Comparison
逐步解耦
```

避免：

```text
重复计算
结果不一致
不必要 runtime
```

---

## 36. Engineering Risk

当前 Global 与 Patch 通过：

```text
list index
```

合并。

由于：

```text
test_paths
```

本身就是由 EXP016 row 顺序构造，因此本实验逻辑正确。

但更稳健的 Benchmark 实现应使用：

```text
relative_path
```

作为唯一 key 进行 merge。

这样即使未来 CSV 顺序变化，也不会产生静默错配。

---

## 37. Limitations

### Limitation 1 - Not a Single-variable Ablation

多项设计同时变化：

```text
layer
feature dimension
spatial aggregation
normal model
nearest-neighbor strategy
image aggregation
```

因此不能直接确定性能差异的唯一原因。

---

### Limitation 2 - Image-Level Evaluation Only

主要评价：

```text
AUROC
AP
```

仍然没有完整：

```text
Pixel AUROC
PRO
IoU
Dice
```

因此不能比较两种方法的 pixel-level localization quality。

---

### Limitation 3 - No Formal Threshold

EXP017 仍使用 ranking metrics。

尚不能正式定义：

```text
FP
FN
Precision
Recall
F1
```

需要独立 Validation Normal 进行 Threshold Calibration。

---

### Limitation 4 - Global Prior Baseline Not Exactly Reproduced

Global：

```text
previous ≈ 0.973016
current  = 0.974603
```

需要后续 Reproducibility Check。

---

## 38. Main Conclusions

### Conclusion 1

Global Baseline 在完整 Bottle image-level anomaly detection 上整体优于当前 Patch Coreset100。

```text
Global:
AUROC = 0.974603

Patch:
AUROC = 0.947619
```

---

### Conclusion 2

Global 对：

```text
broken_large
contamination
```

具有明显优势。

---

### Conclusion 3

Global 与 Patch 在：

```text
broken_small
```

上基本相当。

Patch 数值略高，但只相差约一个 pair ranking。

因此不能宣称存在明显优势。

---

### Conclusion 4

“保留 Patch Spatial Information”并不自动意味着更高的 image-level AUROC。

---

### Conclusion 5

Global Pooling 虽然丢失空间信息，但对会引起整体 representation shift 的 anomaly 仍然非常有效。

---

### Conclusion 6

Patch Baseline 的主要额外价值是：

```text
spatial localization
anomaly map
local failure analysis
```

因此方法评价必须同时考虑：

```text
classification performance
+
localization capability
```

---

### Conclusion 7

稳定 Failure Candidates：

```text
Common High-score Normal:
good/006

Patch-specific High-score Normal:
good/010

Common Low-score Defect:
contamination/003
```

这些样本应进入后续 Failure Analysis。

---

## 39. New Questions

1. Global / Patch 在固定 threshold 下分别有多少 FP / FN？
2. 如何避免使用 Test Label 调 threshold？
3. P90 / P95 / P99 Normal Quantile Threshold 会如何改变 Recall 与 FPR？
4. 哪类 defect 会随着 threshold 提高最先被漏掉？
5. Patch contamination weakness 是 feature layer、memory、aggregation 还是 representation 本身导致？
6. Global 与 Patch 是否可以通过 controlled ablation 拆解真正影响因素？
7. Patch Coreset100 的性能是否对 Reference Split / Sampling Seed 敏感？

---

## 40. Next Step

下一实验：

```text
EXP018
Leakage-Safe Threshold Calibration
+
Operating Point Analysis
```

实验协议：

```text
209 train/good
↓
Reference Normal
+
Validation Normal

Reference
↓
Build Normal Model

Validation
↓
P90 / P95 / P99 Threshold

Independent Test
↓
TP / FP / TN / FN
Precision / Recall / FPR / F1
```

重点避免：

```text
Test Leakage
```

并第一次正式定义：

```text
False Positive
False Negative
```

---

## 41. Status

```text
Global Feature Extraction                 PASS

Normal Center                             PASS

83-image Global Evaluation                PASS

Patch Result Reuse                        PASS

Overall AUROC / AP                        PASS

Per-defect Comparison                     PASS

Failure Candidate Comparison              PASS

Prior Global Baseline Approx Reproduction PASS

Exact Reproduction                        NOT CONFIRMED

Comparison Confound Identified             PASS

Raw-score Cross-method Comparison Avoided PASS

EXP017                                    PASS

Anomaly Detection Unit                    CONTINUE

Git                                       暂不提交
```