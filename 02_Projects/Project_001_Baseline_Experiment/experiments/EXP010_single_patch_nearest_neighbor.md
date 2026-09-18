# EXP010 - Single Patch Nearest Neighbor

## 1. Goal

本实验的目标是：

给定一个 Query Patch，在 EXP009 构建的 Normal Patch Memory Bank 中寻找距离最近的正常 Patch，并将最近邻距离作为最小版本的 Patch Anomaly Score。

核心公式：

$$
s(q)=\min_{m \in M}\|q-m\|_2
$$

其中：

```text
q:
Query Patch

M:
Normal Patch Memory Bank

m:
Memory Bank 中的某个 Normal Patch

s(q):
Query Patch 的 Anomaly Score
```

本实验分为两个阶段：

```text
Experiment A:
Known Training Patch Sanity Check

Experiment B:
Unseen Normal Patch Active Modification
```

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

Feature Layer:
layer2

Device:
CPU

Dataset:
MVTec AD

Category:
bottle
```

---

## 3. Input

Normal Patch Memory Bank：

```text
[163856, 128]
```

单个 Query Patch：

```text
[128]
```

Layer2 Feature Map：

```text
[128, 28, 28]
```

每张图片：

```text
28 × 28 = 784 patches
```

Distance Metric：

```text
Euclidean Distance
L2 Distance
```

---

## 4. Minimum Theory

Memory Bank：

$$
M \in \mathbb{R}^{163856 \times 128}
$$

Query：

$$
q \in \mathbb{R}^{128}
$$

对 Memory Bank 中每一个正常 Patch：

$$
m_i \in \mathbb{R}^{128}
$$

计算：

$$
d_i=\|q-m_i\|_2
$$

展开：

$$
d_i=
\sqrt{
\sum_{j=1}^{128}
(q_j-m_{ij})^2
}
$$

所有距离组成：

$$
D \in \mathbb{R}^{163856}
$$

最终定义：

$$
s(q)=\min(D)
$$

解释：

```text
Minimum Distance 小
↓
Query 能找到相似的正常 Patch
↓
更接近正常模式
```

反之：

```text
Minimum Distance 大
↓
最近的正常 Patch 仍然与 Query 相差较大
↓
异常倾向更高
```

目前这个距离只表示异常倾向。

在没有 Score Distribution 和 Threshold 前，不能仅凭一个数值直接完成 Normal / Anomaly 分类。

---

# Part A - Known Training Patch Sanity Check

## 5. Experiment A Goal

首先使用一个已经存在于 Memory Bank 中的训练 Patch。

Query Image：

```text
bottle/train/good/000.png
```

选择空间位置：

```text
row = 10
col = 5
```

Feature Map Width：

```text
28
```

因此：

$$
patch\_index
=
row \times width + col
$$

$$
=10 \times 28+5
$$

$$
=285
$$

因为该训练图片参与了 Memory Bank 构建，所以理论上：

```text
Query Patch
=
memory_bank[285]
```

这是一个具有已知正确答案的 Sanity Check。

---

## 6. Experiment A Prediction

预测：

```text
Nearest Memory Index:
285

Minimum Distance:
0

Maximum Absolute Feature Error:
0
```

理论依据：

$$
q=m_{285}
$$

因此：

$$
\|q-m_{285}\|_2=0
$$

---

## 7. Core Code

### 7.1 Extract Query Patch

```python
query_patch = patches[
    0,
    patch_index,
    :
]
```

得到：

```text
[128]
```

---

### 7.2 Calculate Difference

```python
difference = (
    memory_bank
    - query_patch.unsqueeze(0)
)
```

Shape：

```text
memory_bank:
[163856, 128]

query_patch.unsqueeze(0):
[1, 128]
```

PyTorch Broadcasting：

```text
[163856, 128]
-
[1, 128]

↓

[163856, 128]
```

相当于：

```text
memory_bank[0]      - query
memory_bank[1]      - query
memory_bank[2]      - query
...
memory_bank[163855] - query
```

---

### 7.3 Calculate Euclidean Distance

```python
distances = torch.norm(
    difference,
    p=2,
    dim=1,
)
```

Shape：

```text
[163856, 128]

↓

[163856]
```

也就是：

```text
一个 Normal Patch
→ 一个 Distance
```

最终得到 163856 个距离。

---

### 7.4 Find Nearest Neighbor

```python
min_distance, nearest_index = torch.min(
    distances,
    dim=0,
)
```

其中：

```text
nearest_index:
哪个 Normal Patch 最近

min_distance:
最近的 Normal Patch 有多近
```

当前：

```text
min_distance
```

就是最小版本的 Patch Anomaly Score。

---

## 8. Experiment A Result

Memory Bank：

```text
Memory bank shape:
torch.Size([163856, 128])
```

Query：

```text
Query image:
bottle/train/good/000.png

Feature map size:
28 x 28

Spatial position:
(10, 5)

Patch index:
285

Query patch shape:
torch.Size([128])
```

Difference：

```text
Difference shape:
torch.Size([163856, 128])
```

Distance：

```text
Distance vector shape:
torch.Size([163856])
```

Nearest Neighbor：

```text
Nearest memory index:
285

Minimum distance:
0.0000000000

Maximum absolute feature error:
0.0
```

程序验证：

```text
Index prediction correct:
True

Distance prediction correct:
True
```

---

## 9. Experiment A Prediction vs Result

### Nearest Index

预测：

```text
285
```

实际：

```text
285
```

结果：

```text
PASS
```

### Minimum Distance

预测：

```text
0
```

实际：

```text
0.0000000000
```

结果：

```text
PASS
```

### Maximum Absolute Feature Error

预测：

```text
0
```

实际：

```text
0.0
```

结果：

```text
PASS
```

---

## 10. Experiment A Analysis

该实验同时验证了：

```text
Dataset Image Order
↓
Feature Extraction
↓
Patch Mapping
↓
Memory Bank Order
↓
Euclidean Distance
↓
Nearest Neighbor
```

整个链路的一致性。

因为 Query Patch 本身存在于 Memory Bank：

$$
q=m_{285}
$$

所以：

$$
s(q)=0
$$

如果这个实验无法得到：

```text
nearest_index = 285
distance = 0
```

则说明以下模块中至少存在一个问题：

```text
Patch Extraction
Memory Bank Mapping
Data Order
Distance Calculation
Nearest Neighbor Search
```

---

# Part B - Unseen Normal Patch Active Modification

## 11. Active Modification

只修改一个变量。

原 Query：

```text
bottle/train/good/000.png
```

修改为：

```text
bottle/test/good/000.png
```

以下条件全部保持不变：

```text
ResNet18
layer2
Normal Memory Bank
row = 10
col = 5
patch_index = 285
Euclidean Distance
```

因此这是一次 Controlled Experiment。

---

## 12. Experiment B Prediction

因为：

```text
test/good/000.png
```

没有参与 Memory Bank 构建，所以不存在训练阶段的 Exact Self-Match。

预测：

```text
Minimum Distance > 0
```

同时：

```text
Nearest Memory Index
不再必然等于 285
```

因为 Query 仍然来自正常图片，我们希望它能够在 Memory Bank 中找到一个相似的正常局部 Feature。

但是当前还没有：

```text
Normal Score Distribution
Anomaly Score Distribution
Validation Threshold
```

因此不能根据单个 Distance 直接分类为 Normal 或 Anomaly。

---

## 13. Experiment B Result

Query：

```text
Query image:
bottle/test/good/000.png

Feature map size:
28 x 28

Spatial position:
(10, 5)

Patch index:
285

Query patch shape:
torch.Size([128])
```

Query Feature 前 10 个数：

```text
tensor([
    0.0000,
    0.1255,
    0.0000,
    0.1023,
    0.0734,
    0.0000,
    0.0000,
    0.2067,
    0.0000,
    0.5317
])
```

Difference：

```text
Difference shape:
torch.Size([163856, 128])
```

Distance：

```text
Distance vector shape:
torch.Size([163856])
```

Nearest Neighbor：

```text
Nearest memory index:
93581

Minimum distance:
1.3316527605

Maximum absolute feature error:
0.5287296175956726
```

---

## 14. Verification Script Issue

本轮程序最后的 `Prediction Verification` 仍然保留了 Experiment A 的旧条件：

```text
Nearest memory index = 285
Minimum distance = 0
```

因此程序显示：

```text
Index prediction correct:
False

Distance prediction correct:
False
```

这两个 `False` **不代表 Nearest Neighbor 算法失败**。

实际原因是：

```text
Query 已经修改
但 Verification 条件没有同步修改
```

因此这是实验脚本验证逻辑的问题，而不是 Feature Extraction 或 Nearest Neighbor 计算错误。

对于 Experiment B，正确的核心验证应该是：

```text
Minimum Distance > 0
```

实际：

```text
1.3316527605 > 0
```

因此符合本轮预测。

---

## 15. Experiment B Prediction vs Result

### Prediction 1

预测：

```text
Minimum Distance > 0
```

实际：

```text
1.3316527605
```

结果：

```text
PASS
```

### Prediction 2

预测：

```text
不存在训练样本中的 Exact Self-Match
```

实际：

```text
nearest_index = 93581
minimum_distance = 1.3316527605
```

结果：

```text
PASS
```

---

## 16. Nearest Neighbor Index Decoding

每张训练图片贡献：

```text
784 patches
```

Nearest Memory Index：

```text
93581
```

训练图片 Index：

$$
93581 // 784 = 119
$$

该图片内部 Patch Index：

$$
93581 \bmod 784 = 285
$$

然后：

$$
row = 285 // 28 = 10
$$

$$
col = 285 \bmod 28 = 5
$$

因此最近邻可以解码为：

```text
Training Image Index:
119

Patch Index:
285

Spatial Position:
(10, 5)
```

也就是说：

```text
test/good/000.png
(row=10, col=5)
```

在全部正常 Memory Bank 中找到的最近邻，恰好来自另一张正常训练图片的：

```text
(row=10, col=5)
```

---

## 17. Analysis

### 17.1 为什么 Unseen Good Patch Distance 不再为 0

对于 Experiment A：

```text
train/good/000.png
```

本身参与 Memory Bank 构建。

因此 Query 可以找到自己：

```text
distance = 0
```

对于 Experiment B：

```text
test/good/000.png
```

没有参与 Memory Bank 构建。

即使它属于正常图片，也不会与 Memory Bank 中某个 Feature 完全相同。

因此：

```text
Minimum Distance:
1.3316527605
```

是合理现象。

---

### 17.2 Spatial Alignment Observation

本实验出现一个值得记录的现象：

```text
Query:
test image
(row=10, col=5)

Nearest Neighbor:
training image index 119
(row=10, col=5)
```

说明在当前 MVTec `bottle` 数据中，不同图片之间具有较稳定的空间对齐。

因此对应空间区域可能拥有相似的局部 Feature。

需要注意：

> 这是本实验观察到的现象，不代表所有工业视觉数据都具有如此稳定的空间对齐。

---

### 17.3 Euclidean Distance 与 Maximum Absolute Error

Minimum Euclidean Distance：

```text
1.3316527605
```

综合考虑全部 128 个 Feature Dimension。

而：

```text
Maximum absolute feature error:
0.5287296175956726
```

只表示 128 个维度中差异最大的单一 Feature Dimension。

因此：

```text
Minimum Euclidean Distance:
当前 Patch Anomaly Score

Maximum Absolute Feature Error:
主要用于 Debug / Verification
```

---

### 17.4 当前不能设置 Normal / Anomaly Threshold

现在只有一个 unseen normal Patch：

```text
score = 1.3316527605
```

我们目前无法判断：

```text
1.33 是高
还是
1.33 是低
```

因为还缺少：

```text
大量 Normal Patch Scores
大量 Defect Patch Scores
Score Distribution
Threshold Selection
```

因此当前正确描述是：

> `test/good/000.png` 的 `(10,5)` Patch，其 nearest-neighbor anomaly score 为 `1.3316527605`。

不能直接称其为 True Negative、False Positive 等分类结果。

---

## 18. Conclusion

EXP010 成功建立了：

```text
Query Patch
↓
Normal Patch Memory Bank
↓
Euclidean Distance
↓
Nearest Neighbor
↓
Minimum Distance
↓
Patch Anomaly Score
```

Experiment A 使用训练集已知 Patch，验证：

```text
nearest_index = 285
distance = 0
```

证明整个 Feature 和 Nearest Neighbor 链路正确。

Experiment B 将 Query 修改为 unseen normal sample：

```text
nearest_index = 93581
minimum_distance = 1.3316527605
```

证明 Nearest Neighbor 可以对未见过的局部 Patch 产生非零 Anomaly Score。

当前已经完成从：

```text
Normal Patch Memory Bank
```

到：

```text
Single Patch Anomaly Score
```

的最小闭环。

---

## 19. New Questions

1. 一张图片的 784 个 Patch 如何一次性计算异常分数？
2. 正常图片中的不同 Patch Score 分布是什么样？
3. `broken_small` 的缺陷区域是否会产生更大的 NN Distance？
4. 如何把 `[784]` 恢复为 `[28,28]`？
5. Image-level Score 应该使用 `max`、`mean` 还是 `top-k`？
6. 如何避免构建非常大的完整 Pairwise Distance Matrix？
7. Threshold 应如何从 Validation Data 中选择？

---

## 20. Next Step

下一实验：

```text
EXP011 - Full Image Patch Scoring
```

目标：

```text
1 Test Image
↓
784 Query Patches
↓
Nearest Neighbor Search
↓
784 Patch Anomaly Scores
```

得到：

```text
[784]
```

随后恢复：

```text
[28, 28]
```

为后续 Anomaly Map Visualization 做准备。

---

## Status

```text
Experiment A:
PASS

Experiment B:
PASS

EXP010:
PASS
```