# EXP014 - Minimal K-Center Coreset Experiment

## 1. Goal

本实验的目标是：

在真实 MVTec Bottle Normal Patch Feature 中构造一个最小 K-center Coreset 实验，理解 Coreset Selection 的基本原理，并比较：

```text
Random Sampling
vs
K-center Greedy Sampling
```

在相同 Memory Capacity 下对 Normal Feature Space 的覆盖能力。

核心问题：

1. 为什么 Random Sampling 可能丢失稀有正常模式？
2. K-center Greedy 如何选择代表性 Patch？
3. 如何理解：

\[
\arg\max_x\min_{c\in C}\|x-c\|_2
\]

4. K-center 是否能改善 Normal Feature Space 的 Tail / Worst-case Coverage？
5. Mean、P95、Maximum Coverage Distance 分别代表什么？
6. K-center 是否一定在所有 Coverage Metric 上优于 Random Sampling？

本实验重点是：

```text
理解 K-center Coreset 的选择机制
```

而不是直接进行完整 anomaly detection evaluation。

---

## 2. Environment

```text
Project:
Project_001_Baseline_Experiment

Conda Environment:
ai_base

Framework:
PyTorch

Backbone:
ResNet18

Pretrained Weights:
ImageNet

Feature Layer:
layer2

Device:
CPU

Dataset:
MVTec AD

Category:
bottle

Random Seed:
42
```

---

## 3. Input

完整 Normal Memory Bank：

```text
torch.Size([163856, 128])
```

从 Full Memory Bank 中固定随机抽取：

```text
Candidate Pool:
[5000, 128]
```

然后分别选择：

```text
Random Selected:
[100, 128]

K-center Selected:
[100, 128]
```

Selection Ratio：

\[
\frac{100}{5000}=2\%
\]

因此这是一个：

```text
5000 Normal Candidate Patches
↓
保留 100 个代表 Patch
```

的最小 Coreset 实验。

---

## 4. Why Coreset

EXP013 已经观察到：

```text
Memory Bank 缩小
↓
Normal Feature Coverage 下降
↓
正常 Patch 更难找到合适最近邻
↓
Anomaly Score 被整体抬高
```

Random Sampling 的潜在问题：

```text
常见 Normal Pattern:
可能重复抽到很多

稀有 Normal Pattern:
可能完全没有抽到
```

因此需要研究：

> 是否可以使用更加有策略的选择方法，在相同 Memory Size 下尽量覆盖完整 Normal Feature Space。

K-center Greedy 就是一个最小解决方案。

---

## 5. Minimum Theory

对于 Normal Candidate：

\[
x_i
\]

以及已经选择的 Coreset：

\[
C=\{c_1,c_2,\ldots,c_k\}
\]

首先定义 Candidate 到 Coreset 的 Coverage Distance：

\[
d(x_i,C)
=
\min_{c\in C}
\|x_i-c\|_2
\]

含义：

> 对于某一个 Candidate Patch，在当前已选择的所有代表 Patch 中，找到离它最近的一个。

如果：

```text
distance small
```

说明该正常模式已经被较好覆盖。

如果：

```text
distance large
```

说明该正常模式目前缺少代表。

---

## 6. K-Center Selection Rule

K-center 每一步选择：

\[
x_{next}
=
\arg\max_x
\min_{c\in C}
\|x-c\|_2
\]

可以分成两步理解。

### Step 1 - Min

对于每个 Candidate：

```text
找到它离当前 Coreset 最近的 Center
```

即：

\[
\min_{c\in C}
\|x-c\|_2
\]

### Step 2 - Max

然后：

```text
在所有 Candidate 中
找到“最近距离”最大的那个
```

即：

\[
\arg\max_x
\]

这个 Patch 就是当前：

```text
Worst-covered Normal Patch
```

因此 K-center 的核心思想可以总结为：

> 每次选择当前覆盖最差的 Normal Patch，让它自己成为新的代表。

---

## 7. Coverage Radius

定义：

\[
R(C)
=
\max_x
\min_{c\in C}
\|x-c\|_2
\]

表示：

> 整个 Normal Candidate Pool 中，当前覆盖最差的 Patch 距离最近 Center 有多远。

K-center 的主要目标是降低：

```text
Worst-case Coverage Distance
```

即 Coverage Radius。

---

## 8. Prediction

Candidate Normal Patches：

```text
5000
```

Selected Patches：

```text
100
```

Selection Ratio：

```text
2%
```

运行前预测：

### Prediction 1

两种选择结果 Shape 都应为：

```text
[100,128]
```

### Prediction 2

预计 K-center Coreset 的 Mean Coverage Distance 低于 Random Sampling。

### Prediction 3

预计 K-center 的 P95 Coverage Distance 低于 Random Sampling。

### Prediction 4

预计 K-center 的 Maximum Coverage Distance 明显低于 Random Sampling。

### Prediction 5

随着 K-center Center 数不断增加，Coverage Radius 应：

```text
单调下降或保持
```

不应该上升。

---

## 9. Core Code

### 9.1 Candidate Pool

```python
permutation = torch.randperm(
    memory_bank.shape[0],
    generator=generator,
)

candidate_indices = permutation[
    :candidate_size
]

candidates = memory_bank[
    candidate_indices
]
```

得到：

```text
[5000,128]
```

---

## 10. Random Baseline

Random Baseline：

```python
random_order = torch.randperm(
    candidate_size,
    generator=generator,
)

random_indices = random_order[
    :selected_size
]

random_selected = candidates[
    random_indices
]
```

结果：

```text
[100,128]
```

Random Sampling 不考虑：

```text
Feature Space Coverage
```

只按照随机索引进行选择。

---

## 11. K-Center Initialization

为了使 Random 和 K-center 对比更干净：

```python
first_index = (
    random_indices[0].item()
)
```

K-center 使用 Random Baseline 的第一个 Patch 作为相同起点。

本实验：

```text
First Index:
3937
```

初始化：

```python
selected_indices = [
    first_index
]
```

然后计算所有 Candidate 到第一个 Center 的距离：

```python
min_distances = torch.cdist(
    candidates,
    first_center,
    p=2,
).squeeze(1)
```

Shape：

```text
[5000]
```

其中：

```text
min_distances[i]
```

表示 Candidate i 当前离 Coreset 最近的距离。

---

## 12. Select Worst-Covered Patch

核心代码：

```python
next_index = torch.argmax(
    min_distances
).item()
```

如果：

```text
min_distances =
[0.5, 1.2, 4.8, 2.0]
```

则：

```text
torch.argmax(...)
=
2
```

说明 Candidate 2 当前距离所有已选代表最远。

因此将它加入 Coreset。

---

## 13. Incremental Distance Update

加入新 Center 后：

```python
new_distances = torch.cdist(
    candidates,
    new_center,
    p=2,
).squeeze(1)
```

然后：

```python
min_distances = torch.minimum(
    min_distances,
    new_distances,
)
```

对于每个 Candidate：

\[
d_i^{new}
=
\min
\left(
d_i^{old},
d(x_i,c_{new})
\right)
\]

因此不需要每次重新计算 Candidate 到所有历史 Center 的距离。

这是一种：

```text
Incremental Update
```

---

## 14. Why Coverage Radius Cannot Increase

加入新的 Center 后：

```text
旧的 Center 没有消失
新的 Center 又增加了一个选择
```

因此对于任意 Candidate：

\[
d(x,C_{k+1})
\le
d(x,C_k)
\]

所以：

\[
R(C_{k+1})
\le
R(C_k)
\]

Coverage Radius 理论上只能：

```text
下降
或保持
```

不能增加。

---

## 15. K-Center Selection Result

实际输出：

```text
Selected   2/100 | radius = 6.465229
Selected   3/100 | radius = 6.180748
Selected   4/100 | radius = 6.030336
Selected   5/100 | radius = 5.959612
Selected   6/100 | radius = 5.650043
Selected   7/100 | radius = 5.618164
Selected   8/100 | radius = 5.406493
Selected   9/100 | radius = 5.402905
Selected  10/100 | radius = 5.364206

Selected  20/100 | radius = 4.963319
Selected  30/100 | radius = 4.495583
Selected  40/100 | radius = 4.245232
Selected  50/100 | radius = 4.070633
Selected  60/100 | radius = 3.892918
Selected  70/100 | radius = 3.772961
Selected  80/100 | radius = 3.630397
Selected  90/100 | radius = 3.550395
Selected 100/100 | radius = 3.450252
```

可以看到 Coverage Radius：

```text
持续单调下降
```

符合数学预测。

---

## 16. K-Center Runtime

```text
K-center Selection Time:
0.104 seconds
```

注意：

本实验只处理：

```text
5000 candidates
100 selected centers
```

因此该时间不能直接外推到完整：

```text
163856 Normal Patches
```

上的大规模 Coreset Selection。

---

## 17. Coverage Evaluation

对于每一个 Candidate：

```python
distances = torch.cdist(
    chunk,
    selected,
    p=2,
)
```

然后：

```python
nearest_distances = (
    distances.min(dim=1).values
)
```

得到：

\[
d(x,C)
=
\min_{c\in C}\|x-c\|_2
\]

最后统计：

```text
Minimum
Mean
P95
Maximum
```

---

## 18. Result - Random 100

Random Sampling：

```text
Minimum distance:
0.000000

Mean distance:
2.119243

P95 distance:
3.423661

Maximum distance:
5.378171
```

---

## 19. Result - K-Center Coreset 100

K-center：

```text
Minimum distance:
0.000000

Mean distance:
2.465316

P95 distance:
3.149299

Maximum distance:
3.450252
```

---

## 20. Direct Comparison

| Metric | Random 100 | K-center 100 | Better |
|---|---:|---:|---|
| Minimum | 0.000000 | 0.000000 | Same |
| Mean | **2.119243** | 2.465316 | Random |
| P95 | 3.423661 | **3.149299** | Coreset |
| Maximum | 5.378171 | **3.450252** | Coreset |

Computed improvements：

```text
Mean coverage improvement:
-0.346074

P95 coverage improvement:
0.274362

Maximum coverage improvement:
1.927919
```

程序判断：

```text
Coreset mean better:
False

Coreset P95 better:
True

Coreset max better:
True
```

---

## 21. Prediction vs Result

### Prediction 1

预测：

```text
Random 和 Coreset
都为 [100,128]
```

结果：

```text
Random:
[100,128]

Coreset:
[100,128]
```

Verdict：

```text
PASS
```

---

### Prediction 2

预测：

```text
Coreset Mean Coverage
<
Random Mean Coverage
```

实际：

```text
Random:
2.119243

Coreset:
2.465316
```

因此：

```text
NOT SUPPORTED
```

这是本实验的重要反例。

---

### Prediction 3

预测：

```text
Coreset P95
<
Random P95
```

实际：

```text
3.149299
<
3.423661
```

Verdict：

```text
PASS
```

---

### Prediction 4

预测：

```text
Coreset Maximum Coverage
<
Random Maximum Coverage
```

实际：

```text
3.450252
<
5.378171
```

Verdict：

```text
PASS
```

Maximum Coverage Distance 降低：

```text
1.927919
```

相对降低约：

```text
35.8%
```

---

### Prediction 5

预测：

```text
Coverage Radius
随着 Center 增加单调不增
```

实际：

```text
6.465229
→
...
→
3.450252
```

Verdict：

```text
PASS
```

---

## 22. Why Random Has Better Mean Coverage

Random Sampling 会按照原始数据分布的密度自然抽样。

如果 Normal Feature Space 中：

```text
某些模式非常常见
```

这些高密度区域更容易被 Random Sampling 多次选中。

因此：

```text
大量常见 Normal Patch
拥有非常接近的 Representative
```

从而降低整体：

```text
Mean Coverage Distance
```

所以 Random：

```text
Mean = 2.119243
```

优于：

```text
K-center Mean = 2.465316
```

并不矛盾。

---

## 23. Why K-Center Has Better Worst-Case Coverage

K-center 并不优先重复覆盖高密度区域。

它反复寻找：

```text
当前最没有被代表的 Normal Patch
```

然后将它加入 Coreset。

因此 Selection Capacity 会更多地分配给：

```text
稀有模式
Feature Space 边缘
低密度区域
Worst-covered regions
```

代价是：

```text
高密度 Normal Region
获得的重复 Representative 更少
```

所以 Mean 可能上升。

但：

```text
Tail Coverage
Worst-case Coverage
```

会明显改善。

---

## 24. P95 Interpretation

Random：

```text
P95 = 3.423661
```

意味着约 95% Candidate 的 Coverage Distance：

```text
<= 3.423661
```

K-center：

```text
P95 = 3.149299
```

说明：

> K-center 不仅改善了最极端的一个 Normal Patch，也改善了整个 Coverage Tail。

因此其作用并不局限于单个 outlier。

---

## 25. Maximum Coverage Interpretation

Random：

```text
Max = 5.378171
```

意味着存在 Normal Candidate：

```text
离最近 Random Representative
仍然有 5.378171
```

K-center：

```text
Max = 3.450252
```

说明整个 Candidate Pool 中：

```text
最差的 Normal Patch
距离最近 Coreset Center
也只有约 3.45
```

因此 K-center 对 Normal Feature Space 的：

```text
Worst-case Coverage
```

明显更好。

---

## 26. Important Correction of Understanding

实验前存在一个过于简单的理解：

```text
Coreset 更有代表性
→
所有 Coverage Metric 都应该更低
```

EXP014 否定了这个理解。

修正后的理解：

```text
K-center 主要优化：

Worst-case Coverage

因此特别关注：

Maximum Coverage Distance

并通常改善：

Tail Coverage / P95

但不保证：

Mean Coverage Distance
```

因此：

> “更有代表性”必须说明依据什么指标。

---

## 27. Relation to Anomaly Detection

正常 Query Patch 应满足：

```text
Normal Query
↓
找到相似 Normal Memory Patch
↓
NN Distance 小
↓
Anomaly Score 小
```

如果 Random Sampling 丢失稀有正常模式：

```text
Rare Normal Query
↓
Memory 中没有对应 Pattern
↓
NN Distance 大
↓
Anomaly Score 被错误抬高
```

这可能导致：

```text
False Positive Risk ↑
```

K-center：

```text
主动覆盖 Feature Space 中
当前最缺少代表的区域
```

因此理论上可能保护：

```text
Rare / Sparse Normal Pattern
```

并减少 Normal Extreme Score。

---

## 28. Limitation

EXP014 只验证：

```text
Normal Feature Space Coverage
```

并没有直接验证：

```text
Image-level AUROC
Good / Defect Separation
False Positive
False Negative
```

因此不能根据 EXP014 单独得出：

```text
K-center 一定提高 anomaly detection accuracy
```

Coverage Objective 与最终 Detection Objective 并不完全等价。

必须将 Coreset 接回 anomaly detection pipeline 验证。

---

## 29. Saved Artifact

```text
results/exp014/
minimal_coreset_experiment.pt
```

包含：

```text
random_seed

candidate_indices
random_indices
coreset_indices

candidate_size
selected_size

random_summary
coreset_summary

coreset_elapsed
```

这些 indices 可供后续 EXP015 精确重建同一批：

```text
Candidate 5000
Random 100
Coreset 100
```

---

## 30. Conclusion

EXP014 成功实现并验证了最小 K-center Greedy Coreset。

核心机制：

```text
Candidate Patch
↓
计算离当前 Coreset 最近距离
↓
找到最近距离最大的 Patch
↓
将它加入 Coreset
↓
更新所有 Candidate 的最近距离
↓
重复
```

核心公式：

\[
\boxed{
x_{next}
=
\arg\max_x
\min_{c\in C}
\|x-c\|_2
}
\]

实验得到：

```text
Random Mean:
2.119243

Coreset Mean:
2.465316
```

说明 K-center 并不保证改善平均 Coverage。

但是：

```text
Random P95:
3.423661

Coreset P95:
3.149299
```

以及：

```text
Random Max:
5.378171

Coreset Max:
3.450252
```

说明 K-center 明显改善：

```text
Tail Coverage
+
Worst-case Coverage
```

因此当前更准确的理解是：

> K-center 的价值主要不是让高密度 Normal Region 的平均距离最小，而是避免 Normal Feature Space 中存在严重没有被代表的区域。

---

## 31. New Questions

1. EXP014 的 Coverage Improvement 能否转化成实际 anomaly detection improvement？
2. Random100 是否会导致正常图片出现更多高 anomaly score？
3. Coreset100 是否能降低 Good Score Upper Tail？
4. Random100 与 Coreset100 的 AUROC 差异如何？
5. Coreset 是否可以在大幅压缩 Memory 的同时接近更大 Memory Bank 的检测性能？
6. K-center 对 outlier 是否过于敏感？
7. 如何将 K-center 扩展到完整 Memory Bank？

---

## 32. Next Step

下一实验：

```text
EXP015 - Random vs Coreset Mini Evaluation
```

比较：

```text
Candidate 5000

Random 100

K-center Coreset 100
```

在：

```text
20 Good
+
20 Broken Small
```

上的 anomaly detection performance。

重点指标：

```text
Good Mean
Good P95
Good Max

Defect Mean
Defect Min

Separation Margin
AUROC
```

---

## Status

```text
K-center Theory:
PASS

argmax(min distance):
PASS

Incremental Update:
PASS

Coverage Radius Verification:
PASS

Random vs Coreset Coverage:
PASS

Prediction Failure Analysis:
PASS

EXP014:
PASS

Coreset Learning Unit:
CONTINUE
```