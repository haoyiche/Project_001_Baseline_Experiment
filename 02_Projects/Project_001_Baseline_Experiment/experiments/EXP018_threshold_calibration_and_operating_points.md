# EXP018 - Leakage-Safe Threshold Calibration and Operating Point Analysis

## 1. Goal

本实验目标是：

将前面仅依赖 AUROC / AP 的 ranking evaluation，推进到真正可执行的：

```text
GOOD / NG decision
```

核心任务包括：

1. 使用 Normal-only Validation Set 校准 anomaly threshold。
2. 避免使用 Test Label 选择 threshold，防止 test leakage。
3. 比较 P90 / P95 / P99 三档 Normal Quantile Threshold。
4. 计算 TP / FP / TN / FN。
5. 计算 Precision / Recall / FPR / Specificity / Accuracy / F1。
6. 分析 threshold 改变后不同 defect type 的 FN 变化。
7. 正式识别 Global 与 Patch 的 FP / FN failure cases。
8. 检查 threshold 参数的理论单调性。
9. 观察 EXP017 → EXP018 Patch AUROC 变化，并识别潜在 stability 问题。

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

Dataset:
MVTec AD

Category:
bottle

Device:
CPU
```

---

## 3. Why EXP018 Is Necessary

EXP016 / EXP017 主要使用：

```text
AUROC
Average Precision
```

这些指标衡量：

```text
ranking quality
```

但实际工业系统最终必须输出：

```text
GOOD
或
NG
```

因此必须定义 threshold：

\[
\hat y(x)=
\begin{cases}
1,&S(x)\ge t\\
0,&S(x)<t
\end{cases}
\]

其中：

```text
0 = good
1 = anomaly
```

---

## 4. Why Test Set Cannot Be Used to Select Threshold

错误做法：

```text
Test Scores
+
Test Labels
↓
遍历 Threshold
↓
选择 Test F1 最大的 Threshold
↓
继续在相同 Test 上报告性能
```

这种做法会产生：

```text
Test Leakage
```

因为 Test Set 已经参与 Model / Hyperparameter Decision。

本实验使用：

```text
Train Good
↓
Reference Normal
+
Validation Normal
```

其中：

```text
Reference
→ 建立 Normal Model

Validation
→ 选择 Threshold

Test
→ 只负责最终评价
```

---

## 5. Train Normal Split

原始：

```text
Bottle Train Good:
209 images
```

固定：

```text
split seed = 2026
```

按照约 80% / 20% 拆分：

```text
Reference Normal:
167

Validation Normal:
42
```

实际检查：

```text
Reference/Validation overlap:
0
```

因此：

```text
No direct sample overlap
```

该检查通过。

---

## 6. Why Both Global and Patch Must Be Rebuilt

### Global

如果使用全部 209 train good 建 Center：

\[
c=
\frac1{209}
\sum_i f_i
\]

再拿其中 42 张做 Validation：

```text
Validation Feature
已经参与 Normal Center 构建
```

存在信息泄漏。

因此必须只使用：

```text
167 Reference Normal
```

建立 Global Center。

---

### Patch

Patch 问题更严重。

如果 Validation Image 的 Patch 已进入 Memory Bank：

```text
query patch
可能找到自己或近似自己
```

会使 NN Distance 人为偏低。

因此 Patch Memory 也必须只由：

```text
167 Reference Normal
```

构建。

---

## 7. Global Reference Model

Reference Images：

```text
167
```

每张提取：

```text
512-D Global Feature
```

Global Normal Center：

\[
c_{ref}
=
\frac1{167}
\sum_{i=1}^{167}f_i
\]

Validation score：

\[
S_G(x)
=
\|f(x)-c_{ref}\|_2
\]

---

## 8. Patch Reference Model

每张 Reference Image：

```text
28 × 28
=
784 patches
```

因此：

\[
167\times784
=
130928
\]

实际 Memory Bank：

```text
torch.Size([130928,128])
```

说明 Patch Memory Construction 与理论一致。

---

## 9. Patch Coreset

从新的 Reference Memory 中：

```text
Candidate Size:
5000

Coreset Size:
100

Seed:
42
```

重新构造：

```text
Coreset100
```

最终：

```text
torch.Size([100,128])
```

K-center Coverage Radius 随 selected center 数量增加持续下降：

```text
2 centers:
6.317871

10 centers:
5.381629

20 centers:
4.672947

50 centers:
4.004524

100 centers:
3.511911
```

符合：

\[
R(C_{k+1})
\le
R(C_k)
\]

K-center 行为正常。

---

## 10. Threshold Definition

Validation Set 只包含正常样本。

对于 normal validation scores：

\[
S_{val}
\]

定义：

\[
t_{90}
=
Q_{0.90}(S_{val})
\]

\[
t_{95}
=
Q_{0.95}(S_{val})
\]

\[
t_{99}
=
Q_{0.99}(S_{val})
\]

直觉：

```text
Percentile ↑
↓
Threshold ↑
↓
更难被判为 anomaly
↓
FP tendency ↓
FN tendency ↑
```

---

## 11. Prediction

### Prediction 1

Threshold 应满足：

```text
P90 <= P95 <= P99
```

---

### Prediction 2

随着 Threshold 上升：

```text
TP 不增
FP 不增

FN 不减
TN 不减
```

因此：

```text
Recall 不增
FPR 不增
```

---

### Prediction 3

P90 相比 P95：

```text
Recall 应更高或相同
FPR 可能更高
```

---

### Prediction 4

P99 相比 P95：

```text
FP 可能减少
但 FN 可能增加
```

---

### Prediction 5

不同 defect type 对 threshold 的敏感性可能不同。

因此必须分析：

```text
broken_large
broken_small
contamination
```

各自的 FN 数量。

---

### Prediction 6

由于 EXP018 使用新的：

```text
167 Reference Split
+
新的 Coreset100
```

其 AUROC 可能与 EXP017 不完全相同。

不能简单把性能变化归因于 threshold。

---

## 12. Validation Score Distribution - Global

Global Validation：

```text
Mean:
3.837120

Max:
6.126794
```

Threshold：

```text
P90:
4.687924

P95:
5.507570

P99:
5.991917
```

满足：

```text
P90 < P95 < P99
```

---

## 13. Validation Score Distribution - Patch

Patch Validation：

```text
Mean:
3.775766

Max:
4.416587
```

Threshold：

```text
P90:
4.130232

P95:
4.214628

P99:
4.405021
```

同样满足：

```text
P90 < P95 < P99
```

---

## 14. Ranking Metrics

在新的 Reference Split 下：

### Global

```text
AUROC:
0.976984

AP:
0.991601
```

### Patch

```text
AUROC:
0.984921

AP:
0.995441
```

需要注意：

这些 AUROC 是在应用具体 P90/P95/P99 threshold 之前计算的。

因此：

```text
Threshold
```

不会导致 AUROC 提升或下降。

---

# 15. Global P90

Threshold：

```text
4.687924
```

Confusion Matrix：

```text
TP = 62
FP = 3
TN = 17
FN = 1
```

Metrics：

```text
Precision:
0.953846

Recall:
0.984127

FPR:
0.150000

F1:
0.968750
```

FN by defect type：

```text
broken_large    0
broken_small    1
contamination   0
```

唯一 FN：

```text
broken_small/007.png
score = 4.218526
```

False Positives：

```text
good/006.png
score = 7.602462

good/008.png
score = 4.842832

good/018.png
score = 4.722548
```

---

# 16. Global P95

Threshold：

```text
5.507570
```

Confusion Matrix：

```text
TP = 52
FP = 1
TN = 19
FN = 11
```

Metrics：

```text
Precision:
0.981132

Recall:
0.825397

FPR:
0.050000

F1:
0.896552
```

FN by defect type：

```text
broken_large    1
broken_small    7
contamination   3
```

False Positive：

```text
good/006.png
score = 7.602462
```

False Negatives：

```text
broken_large/005

broken_small/000
broken_small/007
broken_small/009
broken_small/011
broken_small/013
broken_small/014
broken_small/021

contamination/003
contamination/004
contamination/019
```

---

# 17. Global P99

Threshold：

```text
5.991917
```

Confusion Matrix：

```text
TP = 49
FP = 1
TN = 19
FN = 14
```

Metrics：

```text
Precision:
0.980000

Recall:
0.777778

FPR:
0.050000

F1:
0.867257
```

FN by defect type：

```text
broken_large     1
broken_small    10
contamination    3
```

相比 P95：

```text
FP:
1 → 1

FN:
11 → 14
```

增加的 3 个 FN 全部来自：

```text
broken_small
```

---

# 18. Global Threshold Comparison

| Threshold | TP | FP | TN | FN | Recall | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P90 | 62 | 3 | 17 | 1 | 0.984127 | 0.150000 | 0.968750 |
| P95 | 52 | 1 | 19 | 11 | 0.825397 | 0.050000 | 0.896552 |
| P99 | 49 | 1 | 19 | 14 | 0.777778 | 0.050000 | 0.867257 |

---

# 19. Global Failure Pattern

Global P90 只漏：

```text
broken_small/007
```

当 threshold 提高：

```text
P90 → P95 → P99
```

FN by class：

```text
broken_large:
0 → 1 → 1

broken_small:
1 → 7 → 10

contamination:
0 → 3 → 3
```

因此：

> `broken_small` 是 Global Baseline 最明显的 threshold-sensitive weakness。

尤其：

```text
P95 → P99
```

新增的全部 3 个 FN 均来自：

```text
broken_small
```

---

# 20. Stable Global False Positive

`good/006`：

```text
score = 7.602462
```

在：

```text
P90
P95
P99
```

三种 threshold 下均为 FP。

因此它已经从 EXP017 的：

```text
High-scoring Normal Candidate
```

正式升级为：

```text
Stable Global False Positive
```

说明该样本与 Global Normal Center 的距离异常高。

---

# 21. Patch P90

Threshold：

```text
4.130232
```

Confusion Matrix：

```text
TP = 58
FP = 1
TN = 19
FN = 5
```

Metrics：

```text
Precision:
0.983051

Recall:
0.920635

FPR:
0.050000

F1:
0.950820
```

FN by defect type：

```text
broken_large    0
broken_small    0
contamination   5
```

False Positive：

```text
good/019.png
score = 4.153857
```

False Negatives：

```text
contamination/003.png
3.957147

contamination/004.png
3.963195

contamination/012.png
3.849864

contamination/019.png
4.039086

contamination/020.png
4.050485
```

---

# 22. Patch P95

Threshold：

```text
4.214628
```

Confusion Matrix：

```text
TP = 56
FP = 0
TN = 20
FN = 7
```

Metrics：

```text
Precision:
1.000000

Recall:
0.888889

FPR:
0.000000

F1:
0.941176
```

FN by defect type：

```text
broken_large    2
broken_small    0
contamination   5
```

False Positives：

```text
None
```

False Negatives：

```text
broken_large/003
broken_large/007

contamination/003
contamination/004
contamination/012
contamination/019
contamination/020
```

---

# 23. Patch P99

Threshold：

```text
4.405021
```

Confusion Matrix：

```text
TP = 47
FP = 0
TN = 20
FN = 16
```

Metrics：

```text
Precision:
1.000000

Recall:
0.746032

FPR:
0.000000

F1:
0.854545
```

FN by defect type：

```text
broken_large    5
broken_small    3
contamination   8
```

False Positives：

```text
None
```

---

# 24. Patch Threshold Comparison

| Threshold | TP | FP | TN | FN | Recall | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P90 | 58 | 1 | 19 | 5 | 0.920635 | 0.050000 | 0.950820 |
| P95 | 56 | 0 | 20 | 7 | 0.888889 | 0.000000 | 0.941176 |
| P99 | 47 | 0 | 20 | 16 | 0.746032 | 0.000000 | 0.854545 |

---

# 25. Patch P90 → P95 Trade-off

P90：

```text
FP = 1
FN = 5
```

P95：

```text
FP = 0
FN = 7
```

因此：

```text
P90 → P95
```

收益：

```text
减少 1 个 FP
```

代价：

```text
增加 2 个 FN
```

新增 FN：

```text
broken_large/003
broken_large/007
```

因此：

> 是否选择 P90 或 P95，本质上取决于误杀正常样本与漏检异常样本的业务成本。

---

# 26. Patch P95 → P99 Trade-off

P95：

```text
FP = 0
FN = 7
```

P99：

```text
FP = 0
FN = 16
```

因此：

```text
FP 没有进一步改善
```

但：

```text
FN 增加 9
```

Recall：

```text
0.888889
→
0.746032
```

当前 Test Set 上：

```text
P99
```

没有获得额外 FP 收益，却明显损失 Recall。

---

# 27. Patch Failure Pattern - Low-response Contamination

Patch P90 已经是三档中最低 threshold：

```text
4.130232
```

但仍然存在 5 个 FN：

```text
contamination/003
contamination/004
contamination/012
contamination/019
contamination/020
```

同时：

```text
broken_large FN = 0
broken_small FN = 0
```

因此可以正式确认：

> 当前 Patch Pipeline 对部分 contamination 存在稳定 low-response failure。

这与 EXP016 / EXP017 完全一致。

证据链：

```text
EXP016
↓
Top low-score defects
高度集中 contamination

EXP017
↓
Patch contamination AUROC
明显弱于 Global

EXP018
↓
即使 P90
仍有 5 contamination FN
```

因此：

```text
Failure Mode #1:
Low-response Contamination
```

已经获得多实验支持。

---

# 28. Threshold Limitation vs Model Limitation

如果一个 defect 的 score：

```text
只比 threshold 略低
```

可能通过降低 threshold 改善。

但 Patch P90 中：

```text
contamination/012
score = 3.849864
```

而：

```text
P90 threshold = 4.130232
```

差距明显。

这说明部分 contamination 已深入 Normal Score Distribution。

此时继续降低 threshold：

```text
可能增加大量 FP
```

因此问题不再只是：

```text
threshold selection
```

而更可能涉及：

```text
feature representation
memory representation
feature layer
aggregation
input resolution
```

---

# 29. Threshold Monotonicity Verification

Global：

```text
TP90 >= TP95 >= TP99:
True

FP90 >= FP95 >= FP99:
True

FN90 <= FN95 <= FN99:
True

Recall90 >= Recall95 >= Recall99:
True

FPR90 >= FPR95 >= FPR99:
True
```

Patch：

```text
TP90 >= TP95 >= TP99:
True

FP90 >= FP95 >= FP99:
True

FN90 <= FN95 <= FN99:
True

Recall90 >= Recall95 >= Recall99:
True

FPR90 >= FPR95 >= FPR99:
True
```

因此所有核心 threshold parameter predictions 均通过验证。

---

# 30. Why Precision Is Not Monotonic

Global：

```text
P95 Precision:
0.981132

P99 Precision:
0.980000
```

虽然 threshold 提高，Precision 反而略微下降。

原因：

```text
FP:
1 → 1

TP:
52 → 49
```

Precision：

\[
Precision=
\frac{TP}{TP+FP}
\]

因此 threshold ↑ 并不保证：

```text
Precision ↑
```

同样：

```text
F1
Accuracy
```

也不具有简单单调关系。

---

# 31. Current Test Operating Points

当前六组 Operating Point：

| Method | Threshold | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| Global | P90 | 98.41% | 15% | 0.968750 |
| Global | P95 | 82.54% | 5% | 0.896552 |
| Global | P99 | 77.78% | 5% | 0.867257 |
| Patch | P90 | 92.06% | 5% | 0.950820 |
| Patch | P95 | 88.89% | 0% | 0.941176 |
| Patch | P99 | 74.60% | 0% | 0.854545 |

这些结果代表不同风险偏好：

### Global P90

```text
Very High Recall
但较高 observed FPR
```

### Patch P90

```text
High Recall
+
少量 observed FP
```

### Patch P95

```text
0 observed FP
+
较低 Recall
```

因此不能仅用：

```text
F1
```

自动决定最终工业 threshold。

---

# 32. Important Statistical Limitation - Only 20 Test Good Images

Test Good：

```text
20
```

因此 FPR 的最小变化单位：

\[
1/20
=
5\%
\]

也就是说：

```text
0 FP = 0%
1 FP = 5%
2 FP = 10%
3 FP = 15%
```

因此：

```text
Patch P95
Test FPR = 0%
```

只能解释为：

> 在当前 20 张 Test Good 中没有观察到 False Positive。

不能解释成：

```text
真实工业 FPR = 0%
```

---

# 33. Important Statistical Limitation - Only 42 Validation Normals

Validation：

```text
42 Normal Images
```

P95 尾部大约对应：

\[
42\times0.05
=
2.1
\]

张样本。

P99：

\[
42\times0.01
=
0.42
\]

因此 P99 已经高度依赖极端 Normal Samples。

例如 Patch：

```text
Validation Max:
4.416587

P99:
4.405021
```

非常接近最大值。

因此：

> P99 在本实验主要用于理解 threshold behavior，而不能被视为稳定的生产级 99th-percentile calibration。

---

# 34. EXP017 vs EXP018 - Global Stability

EXP017：

```text
Global AUROC:
0.974603
```

EXP018：

```text
Global AUROC:
0.976984
```

差异：

```text
约 0.00238
```

整体较稳定。

说明 Reference Split 改变后：

```text
Global Center Baseline
```

ranking behavior 没有发生剧烈变化。

---

# 35. EXP017 vs EXP018 - Patch Stability Problem

EXP017：

```text
Patch AUROC:
0.947619
```

EXP018：

```text
Patch AUROC:
0.984921
```

差异：

\[
0.984921-0.947619
=
0.037302
\]

变化明显。

注意：

这不是 Threshold 造成的。

因为 AUROC 在 threshold application 之前计算。

更可能变化来源：

```text
Reference Split
Candidate Pool
Coreset Composition
Sampling Seed Interaction
Normal Coverage
```

因此目前不能得出：

```text
Patch 稳定性能 = 0.984921
```

也不能把：

```text
0.947619
```

当成固定真值。

---

# 36. Important New Question - Coreset Stability

EXP017 与 EXP018 的差异暴露：

```text
Single Coreset Run
```

可能不足以代表 Patch Baseline 的稳定性能。

因此需要后续：

```text
Repeated Seed Experiment
```

固定：

```text
Reference Split
Test Set
Feature Layer
Candidate Size
Coreset Size
```

只改变：

```text
Candidate / Coreset Sampling Seed
```

统计：

```text
AUROC mean
AUROC std
AUROC min
AUROC max

AP mean/std

Threshold behavior
```

---

# 37. Prediction vs Result

### Prediction 1

预测：

```text
P90 <= P95 <= P99
```

实际：

Global：

```text
4.687924
<
5.507570
<
5.991917
```

Patch：

```text
4.130232
<
4.214628
<
4.405021
```

Verdict：

```text
PASS
```

---

### Prediction 2

预测：

```text
Threshold ↑
TP / FP 不增
FN 不减
Recall / FPR 不增
```

所有 Global / Patch monotonicity checks：

```text
True
```

Verdict：

```text
PASS
```

---

### Prediction 3

预测：

```text
P90 Recall >= P95 Recall
```

实际：

Global：

```text
0.984127
>
0.825397
```

Patch：

```text
0.920635
>
0.888889
```

Verdict：

```text
PASS
```

---

### Prediction 4

预测：

```text
P99 可能减少 FP
但增加 FN
```

实际：

Global：

```text
FP:
1 → 1

FN:
11 → 14
```

Patch：

```text
FP:
0 → 0

FN:
7 → 16
```

本实验中：

```text
P99 没有进一步降低 observed FP
但明显增加 FN
```

Verdict：

```text
PARTIALLY SUPPORTED
```

---

### Prediction 5

预测：

```text
不同 defect type
对 threshold 敏感性不同
```

实际：

Global：

```text
broken_small
对高 threshold 最敏感
```

Patch：

```text
contamination
在低 threshold 下仍持续漏检
```

Verdict：

```text
STRONGLY SUPPORTED
```

---

### Prediction 6

预测：

```text
EXP018 AUROC
可能与 EXP017 不完全一致
```

实际：

Global：

```text
0.974603
→
0.976984
```

Patch：

```text
0.947619
→
0.984921
```

Verdict：

```text
SUPPORTED

并暴露新的
Coreset Stability 问题
```

---

# 38. What Was Truly Learned

本实验不只是获得几个 threshold 数字。

当前已经建立：

### 1. AUROC vs Threshold

```text
AUROC
=
ranking quality

Threshold
=
operating decision
```

AUROC 不直接告诉：

```text
GOOD / NG cutoff
```

---

### 2. Threshold Monotonicity

Threshold ↑：

```text
Predicted Positive Set
只能缩小
```

因此：

```text
TP 不增
FP 不增
FN 不减
TN 不减
Recall 不增
FPR 不增
```

---

### 3. FP / FN Business Meaning

FP：

```text
Good → NG
误杀正常产品
```

FN：

```text
Defect → GOOD
漏掉异常产品
```

两者成本可能完全不同。

---

### 4. Threshold Is a Risk Trade-off

Threshold 不是单纯：

```text
越高越安全
```

而是：

```text
FP Risk
vs
FN Risk
```

之间的业务决策。

---

### 5. Model Limitation vs Threshold Limitation

如果 anomaly score 已深入 Normal Distribution：

```text
仅修改 threshold
```

可能必须牺牲大量 Normal 才能救回异常。

此时应考虑：

```text
representation / model
```

改进，而非只调 threshold。

---

# 39. Failure Modes After EXP018

## Global Failure Mode

```text
Threshold-sensitive broken_small
```

尤其高 threshold 下：

```text
broken_small FN
快速增加
```

---

## Patch Failure Mode

```text
Low-response contamination
```

即使 P90：

```text
仍有 5 contamination FN
```

---

## Stable Global FP

```text
good/006
```

在 P90/P95/P99 下持续误报。

---

## Stable Patch Hard Anomalies

```text
contamination/003
contamination/004
contamination/012
contamination/019
contamination/020
```

在 P90 下已经漏检。

---

# 40. Active Modification

Threshold parameter 主动修改：

```text
P90
P95
P99
```

共三档。

并完成：

```text
先预测
→
再运行
→
检查理论行为
→
解释 deviation / trade-off
```

因此 Threshold 核心参数已达到：

```text
Level C
```

能够解释：

```text
参数是什么
改变后发生什么
为什么发生
不同 defect 如何受影响
```

---

# 41. Limitations

### Limitation 1

Validation Normal：

```text
42
```

规模较小。

P99 estimate 不稳定。

---

### Limitation 2

Test Good：

```text
20
```

FPR resolution：

```text
5% per sample
```

无法精细估计真实低 FPR operating regime。

---

### Limitation 3

没有独立 abnormal validation data。

当前 threshold 基于：

```text
Normal-only Calibration
```

因此不能直接针对：

```text
maximum F1
specific Recall target
specific FN cost
```

进行 supervised threshold optimization。

---

### Limitation 4

Patch EXP017 / EXP018 AUROC 差异较大。

需要 repeated experiment 量化随机性。

---

### Limitation 5

仍然只有 Bottle Category。

尚不能保证结论推广到整个 MVTec AD。

---

# 42. Main Conclusions

### Conclusion 1

成功建立 Leakage-safe Threshold Calibration Pipeline：

```text
Reference
→
Normal Model

Validation
→
Threshold

Test
→
Final Evaluation
```

---

### Conclusion 2

P90 / P95 / P99 的参数行为完全符合理论单调性。

---

### Conclusion 3

Global：

```text
P90
Recall = 98.41%
FPR    = 15%
```

表现为：

```text
高 Recall
但高误杀
```

---

### Conclusion 4

Patch：

```text
P95
Recall = 88.89%
Observed FPR = 0%
```

表现为：

```text
较低误杀
但漏检更多
```

---

### Conclusion 5

Global 的主要 threshold-sensitive weakness：

```text
broken_small
```

---

### Conclusion 6

Patch 的稳定 failure mode：

```text
Low-response contamination
```

---

### Conclusion 7

Threshold 只能改变 decision boundary，不能修复所有 representation weakness。

---

### Conclusion 8

Patch Coreset performance 对实验配置可能具有明显 variance。

需要：

```text
Repeated Seed Stability Experiment
```

---

# 43. Next Step

下一实验：

```text
EXP019
Coreset Stability / Repeated Seed Ablation
```

目标：

固定：

```text
Reference Split
Validation Split
Test Set
Backbone
Feature Layer
Candidate Size = 5000
Coreset Size = 100
```

只改变：

```text
Sampling Seed
```

例如：

```text
42
43
44
45
46
```

统计：

```text
AUROC Mean
AUROC Std
Min / Max

AP Mean / Std

Per-defect AUROC
```

用于回答：

> 当前 Patch Baseline 的性能到底有多稳定？

---

# 44. Status

```text
Leakage-safe Train Split                 PASS

Reference / Validation Overlap = 0       PASS

Global Threshold Calibration            PASS

Patch Threshold Calibration             PASS

P90                                     PASS
P95                                     PASS
P99                                     PASS

Active Modification >= 3                PASS

Prediction → Verification               PASS

TP / FP / TN / FN                       PASS

Precision / Recall / FPR / F1          PASS

Per-defect FN Analysis                  PASS

Threshold Monotonicity                  PASS

Formal FP / FN Identification           PASS

Low-response Contamination              CONFIRMED

Threshold-sensitive Broken Small        IDENTIFIED

Coreset Stability Problem               IDENTIFIED

EXP018                                  PASS

Level 1                                 CONTINUE

Git                                     暂不提交
```