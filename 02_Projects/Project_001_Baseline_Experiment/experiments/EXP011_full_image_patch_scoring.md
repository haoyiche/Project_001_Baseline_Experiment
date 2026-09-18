# EXP011 - Full Image Patch Scoring

## 1. Goal

本实验的目标是：

将 EXP010 中的 **Single Patch Nearest Neighbor** 扩展到整张测试图片，对一张图中的全部 784 个 Patch 分别计算最近正常 Patch 的距离，从而得到完整的 Patch Anomaly Score Map。

核心问题：

1. 如何一次处理整张图片的 784 个 Patch？
2. 如何为每个 Patch 计算 Nearest Neighbor Distance？
3. 如何避免直接构建过大的完整距离矩阵？
4. 如何将 `[784]` Patch Scores 恢复为空间上的 `[28,28]` Score Map？
5. `good` 与 `broken_small` 在局部异常分数上是否存在明显差异？
6. 对局部小缺陷而言，Mean Score、Max Score 和 Top-K Score 哪一种更敏感？

本实验使用一个正常测试样本作为 Control，然后只修改输入图片为 `broken_small`，完成一次 Controlled Experiment。

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
```

---

## 3. Input and Parameters

Normal Patch Memory Bank：

```text
[163856, 128]
```

单张测试图片输入：

```text
[3, 224, 224]
```

Layer2 Feature Map：

```text
[1, 128, 28, 28]
```

转为 Patch Feature 后：

```text
[1, 784, 128]
```

移除 Batch Dimension：

```text
[784, 128]
```

Memory Bank：

```text
163856 normal patches
```

Patch Feature Dimension：

```text
128
```

Distance Metric：

```text
Euclidean Distance
L2 Distance
```

Memory Chunk Size：

```text
4096
```

---

## 4. Minimum Theory

对于测试图片中的第 $i$ 个 Patch：

$$
q_i \in \mathbb{R}^{128}
$$

Normal Memory Bank：

$$
M =
\{m_1,m_2,\ldots,m_N\}
$$

其中：

$$
N = 163856
$$

且：

$$
m_j \in \mathbb{R}^{128}
$$

对于每一个 Query Patch，计算它与所有 Normal Patch 的距离：

$$
d_{ij}=\|q_i-m_j\|_2
$$

然后只保留最近正常 Patch：

$$
s_i=
\min_{m\in M}
\|q_i-m\|_2
$$

其中：

$$
s_i
$$

就是第 $i$ 个 Patch 的 anomaly score。

一张图片共有：

$$
28\times28=784
$$

个 Patch，因此最终得到：

$$
S=
[s_1,s_2,\ldots,s_{784}]
$$

Shape：

```text
[784]
```

再恢复空间结构：

```text
[784]

↓

[28, 28]
```

得到低分辨率 Patch Anomaly Score Map。

---

## 5. Why Chunking Is Needed

如果直接计算：

```python
torch.cdist(
    query_patches,
    memory_bank,
)
```

输入 Shape：

```text
Query Patches:
[784, 128]

Memory Bank:
[163856, 128]
```

完整距离矩阵将是：

```text
[784, 163856]
```

元素数量：

$$
784\times163856
=
128463104
$$

即约 1.28 亿个距离。

仅 `float32` 距离矩阵理论存储就接近：

```text
490 MiB
```

因此本实验采用 Chunking：

```text
Memory Bank
↓
每次只取 4096 patches
↓
计算当前 chunk 距离
↓
保存每个 Query 当前最小距离
↓
继续扫描下一个 chunk
```

Chunking 只改变计算方式，不改变最终最近邻结果。

因此当前算法仍然是：

```text
Exact Nearest Neighbor
```

而不是 Approximate Nearest Neighbor。

---

## 6. Core Code

### 6.1 Extract All Image Patches

```python
feature_map = extract_layer2_feature(
    model,
    x,
)
```

得到：

```text
[1, 128, 28, 28]
```

然后：

```python
patches = feature_map_to_patches(
    feature_map
)
```

得到：

```text
[1, 784, 128]
```

移除 Batch Dimension：

```python
query_patches = patches[0].cpu()
```

最终：

```text
[784, 128]
```

---

### 6.2 Initialize Best Distance

```python
best_distances = torch.full(
    (num_queries,),
    float("inf"),
    dtype=query_patches.dtype,
)
```

初始状态：

```text
Patch 0   → inf
Patch 1   → inf
Patch 2   → inf
...
Patch 783 → inf
```

含义：

尚未搜索任何 Normal Patch，因此当前最佳距离先设置为无穷大。

同时：

```python
best_indices = torch.full(
    (num_queries,),
    -1,
    dtype=torch.long,
)
```

保存每个 Query Patch 当前最近的 Memory Bank Index。

---

### 6.3 Scan Memory Bank by Chunk

```python
for start in range(
    0,
    num_memory_patches,
    memory_chunk_size,
):
```

每次处理：

```text
4096 normal patches
```

例如第一块：

```text
0 → 4096
```

第二块：

```text
4096 → 8192
```

直到：

```text
163840 → 163856
```

---

### 6.4 Calculate Distance Matrix

```python
distances = torch.cdist(
    query_patches,
    memory_chunk,
    p=2,
)
```

例如：

```text
Query:
[784,128]

Memory Chunk:
[4096,128]
```

得到：

```text
[784,4096]
```

其中每一行代表：

```text
一个 Query Patch
与当前 4096 个 Normal Patch
之间的距离
```

---

### 6.5 Find Best Match Inside Current Chunk

```python
chunk_min_distances, chunk_min_indices = (
    torch.min(
        distances,
        dim=1,
    )
)
```

输入：

```text
[784,4096]
```

输出：

```text
chunk_min_distances:
[784]

chunk_min_indices:
[784]
```

即：

每个 Query Patch 在当前 Memory Chunk 中找到一个最近邻。

---

### 6.6 Update Global Best Distance

```python
better_mask = (
    chunk_min_distances
    < best_distances
)
```

如果当前 Chunk 找到的距离比之前更小：

```text
True
```

则更新：

```python
best_distances[better_mask] = (
    chunk_min_distances[better_mask]
)
```

最终扫描完整个 Memory Bank 后：

```text
best_distances[i]
```

就等于：

$$
\min_{m\in M}
\|q_i-m\|_2
$$

---

### 6.7 Local Index → Global Memory Index

```python
global_indices = (
    start
    + chunk_min_indices
)
```

例如：

```text
Chunk start:
8192

Local index:
123
```

对应完整 Memory Bank：

```text
8315
```

这样可以追踪：

```text
Query Patch
↓
Nearest Normal Patch
↓
Memory Bank Global Index
```

---

### 6.8 Patch Scores → Anomaly Map

最终 Patch Scores：

```text
[784]
```

恢复空间：

```python
anomaly_map = patch_scores.reshape(
    height,
    width,
)
```

得到：

```text
[28,28]
```

由于 EXP008 已经验证：

$$
index=row\times W+col
$$

所以该 reshape 可以正确恢复原始 Patch 空间关系。

---

## 7. Prediction - Experiment A

Control Sample：

```text
bottle/test/good/000.png
```

预测：

```text
Layer2 Feature Map:
[1,128,28,28]

Query Patches:
[784,128]

Patch Scores:
[784]

Nearest Indices:
[784]

Anomaly Score Map:
[28,28]
```

因为该图片属于 `good`，但没有参与 Memory Bank 构建，因此：

```text
大部分 Patch Distance > 0
```

同时：

$$
max(score)\ge mean(score)\ge min(score)
$$

当前没有 Threshold，因此正常图片中的高分 Patch 只能称为：

```text
high-scoring normal patch
```

不能称为 False Positive。

---

# Part A - Good Control

## 8. Experiment A Result

Input：

```text
bottle/test/good/000.png
```

Query Patch：

```text
Query patches shape:
torch.Size([784, 128])
```

Feature Map：

```text
28 x 28
```

Patch Scores：

```text
Patch scores shape:
torch.Size([784])
```

Nearest Indices：

```text
Nearest indices shape:
torch.Size([784])
```

Score Statistics：

```text
Minimum score:
0.003383

Mean score:
0.819022

Maximum score:
2.599664
```

Anomaly Map：

```text
torch.Size([28, 28])
```

Highest-Scoring Patch：

```text
Patch index:
444

Spatial position:
(15, 24)

Anomaly score:
2.599664

Nearest memory index:
51460
```

Top-5 Highest-Scoring Patches：

```text
Rank 1
position = (15,24)
score    = 2.599664

Rank 2
position = (3,12)
score    = 2.434475

Rank 3
position = (16,24)
score    = 2.252240

Rank 4
position = (13,2)
score    = 2.118351

Rank 5
position = (12,24)
score    = 2.094882
```

Verification：

```text
Expected patch score shape:
(784,)

Actual:
(784,)

Patch score shape correct:
True
```

```text
Expected anomaly map shape:
(28,28)

Actual:
(28,28)

Anomaly map shape correct:
True
```

---

## 9. Experiment A Analysis

正常测试图并不意味着所有 Patch Score 都接近 0。

即使图片标签为：

```text
good
```

测试样本与训练样本仍然存在正常变化，例如：

```text
位置变化
光照变化
高光
边缘
纹理变化
背景变化
CNN Feature Variation
```

因此出现：

```text
Maximum score:
2.599664
```

并不意味着该 Patch 已经是 False Positive。

目前还没有 Threshold。

准确描述应该是：

> `good/000.png` 中存在若干 high-scoring normal patches。

Top-5 中：

```text
(12,24)
(15,24)
(16,24)
```

具有一定空间聚集现象。

但是在没有 Visualization 和 Ground Truth 对照前，不能进一步判断这些位置为什么得到较高 Score。

---

# Part B - Broken Small Active Modification

## 10. Active Modification

本轮只修改：

```text
Input Image
```

从：

```text
bottle/test/good/000.png
```

修改为：

```text
bottle/test/broken_small/000.png
```

以下所有条件保持不变：

```text
ResNet18
ImageNet pretrained weights
layer2
Input resolution = 224×224
Normal Memory Bank
Patch Feature Dimension = 128
Euclidean Distance
Chunk Size = 4096
Nearest Neighbor Algorithm
```

因此这是一次 Controlled Experiment。

---

## 11. Prediction - Experiment B

由于 `broken_small` 属于局部小缺陷，预测：

### Prediction 1

绝大多数 Patch 仍然来自正常区域。

因此：

```text
Mean Patch Score
不一定明显增加
```

甚至可能接近正常图片。

### Prediction 2

缺陷附近 Patch 与 Normal Memory Bank 中的正常模式差异更大。

因此：

```text
Maximum Patch Score
预计增加
```

### Prediction 3

Top-K 高分 Patch 预计比 `good` 样本更明显。

### Prediction 4

如果 Patch-level representation 能捕捉到局部小缺陷，高分 Patch 应出现一定的空间聚集，而不是完全随机分散。

### Prediction 5

当前没有 Threshold 和 Ground Truth 对照。

因此即使 Score 增大，也暂时不能称为：

```text
TP
FP
FN
```

也不能直接声称：

```text
缺陷定位成功
```

---

## 12. Experiment B Result

Input：

```text
bottle/test/broken_small/000.png
```

Query Patch：

```text
Query patches shape:
torch.Size([784, 128])
```

Patch Scores：

```text
Patch scores shape:
torch.Size([784])
```

Nearest Indices：

```text
Nearest indices shape:
torch.Size([784])
```

Score Statistics：

```text
Minimum score:
0.003654

Mean score:
0.778466

Maximum score:
3.906899
```

Anomaly Map：

```text
torch.Size([28, 28])
```

Highest-Scoring Patch：

```text
Patch index:
483

Spatial position:
(17, 7)

Anomaly score:
3.906899

Nearest memory index:
33988
```

Top-5 Highest-Scoring Patches：

```text
Rank 1
position = (17,7)
score    = 3.906899

Rank 2
position = (18,7)
score    = 3.702626

Rank 3
position = (19,8)
score    = 3.525064

Rank 4
position = (19,7)
score    = 3.482851

Rank 5
position = (19,6)
score    = 3.354582
```

Verification：

```text
Patch score shape correct:
True

Anomaly map shape correct:
True
```

---

## 13. Good vs Broken Small Comparison

| Metric | Good 000 | Broken Small 000 |
|---|---:|---:|
| Minimum Score | 0.003383 | 0.003654 |
| Mean Score | 0.819022 | 0.778466 |
| Maximum Score | 2.599664 | 3.906899 |
| Top-1 Position | (15,24) | (17,7) |
| Top-1 Score | 2.599664 | 3.906899 |

### Mean Score

变化：

$$
0.778466-0.819022
=
-0.040556
$$

相对变化约：

```text
-4.95%
```

因此 `broken_small` 的 Mean Score 并没有增加，反而略微下降。

这与运行前预测一致：

> 局部小缺陷只影响少量 Patch，因此整图平均值可能被大量正常区域稀释。

---

### Maximum Score

变化：

$$
3.906899-2.599664
=
1.307235
$$

相对增加约：

```text
50.3%
```

因此：

```text
Broken Small Max Score
明显高于
Good Max Score
```

符合运行前预测。

---

## 14. Top-5 Score Comparison

Good Top-5：

```text
2.599664
2.434475
2.252240
2.118351
2.094882
```

平均值约：

```text
2.299922
```

Broken Small Top-5：

```text
3.906899
3.702626
3.525064
3.482851
3.354582
```

平均值约：

```text
3.594404
```

相对增加约：

```text
56.3%
```

因此相比 Mean Score：

```text
Top-K Local Score
```

对当前 `broken_small` 局部异常表现出更明显的响应。

---

## 15. Spatial Distribution Analysis

Good Top-5：

```text
(15,24)
(3,12)
(16,24)
(13,2)
(12,24)
```

存在部分：

```text
col = 24
```

聚集，但整体仍较为分散。

Broken Small Top-5：

```text
(17,7)
(18,7)
(19,8)
(19,7)
(19,6)
```

这些位置集中在：

```text
row ≈ 17~19

col ≈ 6~8
```

形成明显的局部空间聚集。

这说明：

> `broken_small/000.png` 中存在一片连续局部区域，其 Patch Feature 与 Normal Memory Bank 中的正常局部模式差异较大。

但是当前尚未加载 Ground Truth Mask，因此不能直接断言：

> 该高分区域就是实际缺陷区域。

目前只能将其记录为：

```text
Spatially clustered high-score region
```

---

## 16. Why Mean Score Can Fail for Small Defects

假设一张图片有：

```text
784 patches
```

其中只有少量 Patch 被小缺陷影响，例如：

```text
正常区域:
大量 patches

缺陷区域:
少量 patches
```

如果使用 Mean：

$$
S_{mean}
=
\frac{1}{784}
\sum_{i=1}^{784}s_i
$$

大量正常 Patch 会稀释少数异常 Patch 的高分。

因此即使存在明显局部异常：

```text
Mean Score
也可能变化很小
```

甚至可能下降。

本次实验实际得到：

```text
Good Mean:
0.819022

Broken Small Mean:
0.778466
```

说明 Mean Score 对当前局部小缺陷并不敏感。

---

## 17. Why Max Score Is More Sensitive

如果定义：

$$
S_{max}
=
\max_i s_i
$$

那么只要存在一个局部区域明显偏离所有正常模式：

```text
Max Score
就会保留该局部异常信号
```

本实验：

```text
Good Max:
2.599664

Broken Small Max:
3.906899
```

Broken Small 相比 Good 提升约：

```text
50.3%
```

因此当前实验支持：

> 对局部小缺陷而言，Max Patch Score 比 Mean Patch Score 更敏感。

---

## 18. Relation to Global Feature Baseline

之前 Global Feature Baseline：

```text
Image
↓
ResNet Feature Map
↓
Global Average Pooling
↓
One Global Feature
```

局部异常在进入最终 Representation 之前，会与大量正常空间区域进行聚合。

因此：

```text
Small Local Defect
↓
Possible Global Dilution
```

当前 Patch-level 方法：

```text
Image
↓
784 Local Patch Features
↓
每个 Patch 独立搜索 Normal Memory Bank
↓
784 Local Anomaly Scores
```

局部高分不会先被全图正常区域平均掉。

因此本实验开始验证当前 Learning Unit 的核心假设：

> Global representation 可能对局部小缺陷不够敏感，而 Patch representation 能够保留局部异常信号。

当前证据：

```text
Mean Score:
没有增加

Max Score:
明显增加

Top-K Score:
明显增加

High-score patches:
出现局部空间聚集
```

---

## 19. Important Limitation

虽然 `broken_small/000.png` 的高分 Patch 出现明显空间聚集：

```text
row 17~19
col 6~8
```

目前尚未验证：

```text
该区域
是否和真实 Ground Truth Defect Region 重合
```

因此当前不能写：

```text
Successfully localized the defect
```

正确结论是：

> Patch-level NN scoring produced a spatially clustered high-score region on the broken-small sample.

下一实验需要加载 MVTec Ground Truth Mask 进行空间对照。

---

## 20. Prediction vs Result

### Prediction 1

预测：

```text
Broken Small Mean Score
不一定明显升高
```

结果：

```text
Good:
0.819022

Broken Small:
0.778466
```

结论：

```text
PASS
```

---

### Prediction 2

预测：

```text
Broken Small Max Score
预计明显增加
```

结果：

```text
Good:
2.599664

Broken Small:
3.906899
```

增长约：

```text
50.3%
```

结论：

```text
PASS
```

---

### Prediction 3

预测：

```text
Top-K Scores
预计增加
```

结果：

```text
Good Top-5 Mean:
2.299922

Broken Small Top-5 Mean:
3.594404
```

增长约：

```text
56.3%
```

结论：

```text
PASS
```

---

### Prediction 4

预测：

```text
高分 Patch 出现空间聚集
```

结果：

```text
(17,7)
(18,7)
(19,8)
(19,7)
(19,6)
```

形成明显局部聚集。

结论：

```text
PASS
```

---

## 21. Conclusion

EXP011 成功将 EXP010 的 Single Patch Nearest Neighbor 扩展为 Full Image Patch Scoring。

完整数据流：

```text
Test Image
↓
ResNet18 layer2
↓
[1,128,28,28]
↓
Patch Features
↓
[784,128]
↓
Chunked Exact Nearest Neighbor Search
↓
[784] Patch Anomaly Scores
↓
reshape
↓
[28,28] Patch Anomaly Score Map
```

本实验同时完成了：

```text
Good Control
vs
Broken Small Active Modification
```

实验结果表明：

```text
Mean Score:
0.819022 → 0.778466
没有体现出小缺陷异常增强

Maximum Score:
2.599664 → 3.906899
明显增加

Top-5 Mean:
2.299922 → 3.594404
明显增加

Broken Small High-score Region:
出现明显空间聚集
```

因此当前证据支持：

> 对局部小缺陷而言，局部 Patch Score，尤其是 Max / Top-K Score，比整图 Patch Mean 更敏感。

同时，这也说明 Patch-level representation 能够保留 Global Average 类聚合可能稀释的局部异常信息。

但是当前尚未与 Ground Truth Mask 对齐，因此还不能正式宣称缺陷定位成功。

---

## 22. New Questions

1. `broken_small` 的高分区域是否真的对应 Ground Truth 缺陷？
2. 如何将 `[28,28]` Anomaly Map 上采样回原图尺寸？
3. 如何把 Anomaly Map 与原图进行 Overlay？
4. Patch Score Map 与 Ground Truth Mask 的空间重合程度如何？
5. Max Score 是否适合所有异常类型？
6. Top-K Mean 是否比 Max 更稳定？
7. layer2 与 layer3 的定位性能有什么差异？
8. 输入分辨率提高后，小缺陷定位是否进一步改善？
9. 如何从单张图片实验扩展到整个 bottle test set？

---

## 23. Next Step

下一实验：

```text
EXP012 - Anomaly Map Visualization and Ground Truth Comparison
```

目标：

```text
28×28 Patch Anomaly Map
↓
Upsample
↓
Original Image Size
↓
Visualization
↓
Ground Truth Mask
↓
Spatial Comparison
```

重点验证：

> `broken_small/000.png` 中 `(row≈17~19, col≈6~8)` 的高分区域是否真实对应缺陷区域。

---

## Status

```text
Experiment A - Good Control:
PASS

Experiment B - Broken Small Active Modification:
PASS

Chunked Exact NN:
PASS

Full Image Patch Scoring:
PASS

28×28 Patch Score Map:
PASS

EXP011:
PASS

Current Learning Unit:
CONTINUE
```