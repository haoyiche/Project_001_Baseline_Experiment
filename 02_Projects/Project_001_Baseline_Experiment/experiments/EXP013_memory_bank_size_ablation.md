# EXP013 - Memory Bank Size Ablation

## 1. Goal

本实验的目标是：

研究 Normal Patch Memory Bank 大小对 Patch Anomaly Score、空间异常结构和计算成本的影响。

比较三种 Memory Bank：

```text
100% Full Memory Bank

10% Random Memory Bank

1% Random Memory Bank
```

核心问题：

1. Memory Bank 缩小后，Nearest Neighbor Score 会如何变化？
2. Reduced Memory Bank 是否还能保留 Full Memory 的空间异常结构？
3. Memory Bank Size 如何影响正常模式覆盖？
4. Memory Reduction 能够减少多少 Pairwise Distance Workload？
5. 10% 或 1% Memory Bank 是否仍能保留 `broken_small` 的主要高分区域？
6. 为什么 PatchCore 需要 Coreset Selection？

本实验使用：

```text
broken_small/000.png
```

作为固定 Query Image。

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

Test Image:
broken_small/000.png

Random Seed:
42
```

---

## 3. Input

Full Normal Patch Memory Bank：

```text
[163856, 128]
```

Full-bank Patch Scores：

```text
[784]
```

来自：

```text
results/exp012/
broken_small_000_scores.pt
```

Query Patches：

```text
[784,128]
```

---

## 4. Experiment Design

本实验使用固定随机排列：

```python
generator = torch.Generator()
generator.manual_seed(42)
```

然后：

```python
permutation = torch.randperm(
    num_memory,
    generator=generator,
)
```

10% Bank：

```python
indices_10 = permutation[:size_10]
```

1% Bank：

```python
indices_1 = permutation[:size_1]
```

由于：

```text
size_1 < size_10
```

且二者来自同一个 permutation：

```text
M_1% ⊂ M_10% ⊂ M_full
```

这是一个 Nested Subset Experiment。

---

## 5. Mathematical Prediction

Patch Anomaly Score：

$$
s(q,M)
=
\min_{m\in M}
\|q-m\|_2
$$

由于：

$$
M_{1\%}
\subset
M_{10\%}
\subset
M_{full}
$$

因此搜索候选越少，最近邻距离只能保持或增大。

对于每一个 Query Patch 都必须满足：

$$
s_{full}
\le
s_{10\%}
\le
s_{1\%}
$$

也就是说：

```text
Reduced Memory Bank
不能产生比 Full Memory Bank 更小的 NN Distance
```

除极小浮点误差外，该关系应对全部 784 个 Patch 成立。

---

## 6. Prediction

### Memory Size

Full：

```text
[163856,128]
```

10%：

```text
约 [16385,128]
```

1%：

```text
约 [1638,128]
```

---

### Score Prediction

预测：

```text
Full score
<=
10% score
<=
1% score
```

对所有 Patch 成立。

---

### Compute Prediction

10% Memory：

```text
约减少 90% Pairwise Distance Comparisons
```

1% Memory：

```text
约减少 99% Pairwise Distance Comparisons
```

---

### Representation Prediction

10% Random Memory：

```text
预计仍能较好保留主要异常空间结构。
```

1% Random Memory：

```text
可能开始明显损失正常模式覆盖，
导致大量正常 Patch Score 被整体抬高。
```

---

## 7. Important Terminology

本实验使用的是：

```text
Random Memory Bank Subsampling
```

不是正式 PatchCore 中的：

```text
Coreset Sampling
```

Random Sampling：

```text
随机保留 Normal Patch
```

Coreset：

```text
有意识选择具有代表性的 Normal Patch，
尽可能覆盖完整 Feature Space。
```

因此当前算法准确描述为：

> Minimal PatchCore-style Patch Baseline with Random Memory-bank Ablation.

---

## 8. Core Code

### 8.1 Reproducible Random Sampling

```python
random_seed = 42

generator = torch.Generator()

generator.manual_seed(
    random_seed
)
```

Random Seed 的作用：

```text
保证每次运行
使用同一组随机 Memory Patch
```

便于实验复现。

`42` 本身没有特殊意义，重点是：

```text
固定
记录
复现
```

---

### 8.2 Build Nested Subsets

```python
permutation = torch.randperm(
    num_memory,
    generator=generator,
)
```

10%：

```python
indices_10 = permutation[
    :size_10
]
```

1%：

```python
indices_1 = permutation[
    :size_1
]
```

因为 1% 使用相同 permutation 的更短前缀：

```text
1% ⊂ 10% ⊂ Full
```

这使得：

$$
score_{full}
\le
score_{10\%}
\le
score_{1\%}
$$

成为可验证的数学关系。

---

### 8.3 Reuse EXP012 Full Scores

```python
saved_data = torch.load(
    full_score_path,
    map_location="cpu",
)
```

再：

```python
full_scores = (
    saved_data["patch_scores"]
    .cpu()
    .float()
)
```

这样不需要重新计算 Full Memory Bank 的：

```text
128,463,104
```

组 Pairwise Distances。

这是实验结果复用。

---

### 8.4 Score Summary

```python
min_score = scores.min()
mean_score = scores.mean()
max_score = scores.max()
```

同时使用：

```python
torch.topk(
    scores,
    k=5,
)
```

提取最高异常 Patch。

---

### 8.5 Mean Absolute Error

Reduced Score 与 Full Score：

```python
difference = (
    reduced_scores
    - full_scores
)
```

MAE：

```python
mae = torch.mean(
    torch.abs(difference)
)
```

含义：

> Reduced Memory Bank 中，每个 Patch Score 平均偏离 Full Reference 多少。

---

### 8.6 Maximum Absolute Difference

```python
max_abs_difference = torch.max(
    torch.abs(difference)
)
```

表示：

> 受 Memory Reduction 影响最大的 Patch，其 Score 改变了多少。

---

### 8.7 Score-map Correlation

```python
correlation_matrix = torch.corrcoef(
    torch.stack(
        [
            full_scores,
            reduced_scores,
        ]
    )
)
```

得到 Pearson Correlation。

Correlation 主要衡量：

```text
哪些空间位置高
哪些空间位置低
```

这种相对空间分数结构是否被保留。

因此：

```text
MAE:
数值变化有多大

Correlation:
空间模式变化有多大
```

二者衡量的不是同一件事。

---

### 8.8 Monotonicity Verification

```python
violations = (
    reduced_scores
    < full_scores - tolerance
).sum()
```

如果 Reduced Bank 是 Full Bank 的子集，则：

```text
Reduced Score < Full Score
```

理论上不应该发生。

因此：

```text
violations = 0
```

是重要的 Sanity Check。

---

### 8.9 Floating-point Tolerance

使用：

```python
tolerance = 1e-5
```

是为了容忍类似：

```text
1.0000000
vs
0.99999994
```

这种浮点数计算误差。

该 tolerance 很小，不能掩盖明显算法错误。

---

### 8.10 Top-5 Spatial Overlap

Full Top-5：

```python
full_top = set(
    full_summary["top_positions"]
)
```

Reduced Top-5：

```python
top_10
top_1
```

交集：

```python
full_top.intersection(
    top_10
)
```

用于测量：

> Full Memory 中最异常的空间位置，有多少仍然出现在 Reduced Bank 的 Top-5 中。

该指标不评价具体排名，只评价 Top-K Membership。

---

### 8.11 Pairwise Distance Workload

计算复杂度：

$$
O(QM)
$$

其中：

```text
Q:
Query Patch Count

M:
Memory Bank Size
```

Query：

```text
784
```

所以：

```python
pairs = (
    num_queries
    * memory_size
)
```

---

## 9. Result - Memory Bank Shapes

Full：

```text
(163856, 128)
```

10%：

```text
(16385, 128)
```

1%：

```text
(1638, 128)
```

---

## 10. Full Memory Reference

Full：

```text
Minimum score:
0.003654

Mean score:
0.778466

Maximum score:
3.906899
```

Top-5：

```text
Rank 1
(17,7)
3.906899

Rank 2
(18,7)
3.702626

Rank 3
(19,8)
3.525064

Rank 4
(19,7)
3.482851

Rank 5
(19,6)
3.354582
```

---

## 11. 10% Random Memory Result

Memory Size：

```text
16385
```

Score：

```text
Minimum:
0.004367

Mean:
0.914699

Maximum:
3.972243
```

Top-5：

```text
Rank 1
(17,7)
3.972243

Rank 2
(19,7)
3.753681

Rank 3
(18,7)
3.702626

Rank 4
(18,8)
3.582522

Rank 5
(19,8)
3.578992
```

Compared with Full：

```text
Mean absolute score difference:
0.136233

Maximum absolute score difference:
0.678517

Score-map correlation:
0.981391

Monotonicity violations:
0
```

NN Search Time：

```text
0.051 s
```

---

## 12. 1% Random Memory Result

Memory Size：

```text
1638
```

Score：

```text
Minimum:
0.005524

Mean:
1.204370

Maximum:
4.248710
```

Top-5：

```text
Rank 1
(17,7)
4.248710

Rank 2
(18,7)
4.066178

Rank 3
(19,7)
3.883898

Rank 4
(19,8)
3.875601

Rank 5
(19,6)
3.800792
```

Compared with Full：

```text
Mean absolute score difference:
0.425904

Maximum absolute score difference:
2.905765

Score-map correlation:
0.893151

Monotonicity violations:
0
```

NN Search Time：

```text
0.005 s
```

---

## 13. Nested Memory Verification

预测：

$$
score_{full}
\le
score_{10\%}
\le
score_{1\%}
$$

实际：

```text
Full <= 10% for every patch:
True

10% <= 1% for every patch:
True
```

因此：

```text
Mathematical Prediction:
PASS
```

全部 784 个 Patch 均符合理论关系。

---

## 14. Pairwise Distance Workload

Full：

```text
128,463,104 patch pairs
```

10%：

```text
12,845,840 patch pairs
```

约减少：

```text
90%
```

1%：

```text
1,284,192 patch pairs
```

约减少：

```text
99%
```

---

## 15. Score Comparison

| Metric | Full | 10% | 1% |
|---|---:|---:|---:|
| Memory Patches | 163856 | 16385 | 1638 |
| Min Score | 0.003654 | 0.004367 | 0.005524 |
| Mean Score | 0.778466 | 0.914699 | 1.204370 |
| Max Score | 3.906899 | 3.972243 | 4.248710 |
| MAE vs Full | 0 | 0.136233 | 0.425904 |
| Correlation vs Full | 1.000000 | 0.981391 | 0.893151 |
| Monotonicity Violations | 0 | 0 | 0 |

---

## 16. Mean Score Analysis

Full：

```text
0.778466
```

10%：

```text
0.914699
```

相对 Full 增加约：

```text
17.5%
```

1%：

```text
1.204370
```

相对 Full 增加约：

```text
54.7%
```

说明：

```text
Memory Bank 越小
↓
正常模式覆盖越少
↓
很多 Query Patch 找不到足够相似的正常邻居
↓
Nearest Neighbor Distance 整体抬高
```

因此 Memory Bank Reduction 不仅影响异常 Patch，也会提高正常区域的 anomaly score。

这可能带来：

```text
False Positive Risk ↑
```

当前尚未设置 Threshold，因此暂时只记录为风险趋势。

---

## 17. Maximum Score Analysis

Full：

```text
3.906899
```

10%：

```text
3.972243
```

约增加：

```text
1.67%
```

1%：

```text
4.248710
```

约增加：

```text
8.75%
```

相比 Mean Score：

```text
17.5%
54.7%
```

Max Score 对 Memory Reduction 相对更稳定。

原因：

真正异常区域本身已经难以在 Full Memory 中找到相似正常模式。

因此：

```text
Full Bank
已经具有较大 NN Distance
```

继续删减 Memory Bank 后，异常区域距离只会有限增加。

而很多正常区域原本依赖某些特定正常邻居，一旦这些正常邻居被随机删除，Score 就可能明显增加。

---

## 18. Score-map Correlation Analysis

10%：

```text
Correlation:
0.981391
```

说明：

> 即使只保留 10% Normal Patch，整体空间 anomaly score pattern 仍然与 Full Memory 非常接近。

即：

```text
Full 中高的位置
通常在 10% 中仍然较高

Full 中低的位置
通常在 10% 中仍然较低
```

1%：

```text
Correlation:
0.893151
```

仍具有明显相关性，但已明显低于 10%。

说明：

```text
1% Memory Bank
开始明显改变整个 Score Map 的空间结构。
```

---

## 19. Top-5 Spatial Stability

结果：

```text
Full vs 10% Top-5 overlap:
4/5

Full vs 1% Top-5 overlap:
5/5
```

1% Top-5 overlap 看起来比 10% 更高。

但不能因此认为：

```text
1% 比 10% 更好
```

因为 Top-5 Overlap 只观察：

```text
最高的五个 Patch
```

而不反映其余：

```text
779 个 Patch
```

的变化。

其他指标显示：

```text
10% Correlation:
0.981391

1% Correlation:
0.893151
```

以及：

```text
10% MAE:
0.136233

1% MAE:
0.425904
```

都说明：

```text
10% 更接近 Full Reference
```

因此 Top-5 Overlap 不能单独作为整体质量指标。

---

## 20. Why Anomaly Region Remains Stable

Full、10%、1% 的最高异常区域均主要集中于：

```text
row ≈ 17~19
col ≈ 6~8
```

即 EXP012 Ground Truth 所在区域。

说明：

> `broken_small` 的真实缺陷信号非常强，即使 Memory Bank 大幅缩小，这些局部 Patch 仍然是整张图片最难与正常模式匹配的区域。

因此：

```text
Defect Ranking
相对稳定
```

但这不代表正常区域 Score 也保持稳定。

---

## 21. Memory Coverage Interpretation

Memory Bank 越大：

```text
更多正常视觉模式
↓
更容易找到相似 Normal Patch
↓
正常区域 NN Distance 更小
↓
Normal Feature Space Coverage 更完整
```

Memory Bank 越小：

```text
正常模式被删除
↓
某些 Query Normal Patch 找不到正确邻居
↓
被迫匹配较差邻居
↓
NN Distance 增加
↓
Anomaly Score 被抬高
```

因此当前实验表明：

> Memory Bank Reduction 的主要风险之一，是损伤 Normal Feature Space Coverage。

---

## 22. 10% Memory Bank Interpretation

10% Random Bank：

```text
Memory:
约降低 90%

Pairwise workload:
约降低 90%

Score-map correlation:
0.981391

MAE:
0.136233
```

对于本样本：

```text
10% Random Memory Bank
保留了较好的空间 anomaly structure
```

同时显著降低计算量。

但因为目前只测试：

```text
broken_small/000.png
```

不能声明：

```text
10% 是整个 bottle 数据集最佳 Memory Size
```

---

## 23. 1% Memory Bank Interpretation

1% Random Bank：

```text
Memory:
约降低 99%

Pairwise workload:
约降低 99%
```

仍然保持：

```text
Full Top-5 overlap:
5/5
```

说明真实缺陷信号仍然很强。

但是：

```text
Mean Score:
0.778466
→
1.204370

Correlation:
0.893151

MAE:
0.425904

Max Absolute Difference:
2.905765
```

说明：

```text
Normal Feature Coverage
已经明显下降
```

因此 1% 最大的问题不是：

```text
无法发现明显缺陷
```

而是：

```text
大量正常区域也开始变得“不够正常”
```

后续可能导致更高 False Positive Risk。

---

## 24. Search Time Interpretation

实际测得：

```text
10%:
0.051 s

1%:
0.005 s
```

二者大约相差：

```text
10×
```

与 Pairwise Comparison 数量约相差：

```text
10×
```

基本一致。

但是：

```text
Full Memory Search
本轮没有重新计时
```

而是直接复用了 EXP012 保存结果。

因此当前不能严格写：

```text
10% 比 Full 实测快 10×
```

只能写：

> 10% 和 1% 本轮 NN 搜索时间分别约为 0.051 s 和 0.005 s；理论 Pairwise Workload 相比 Full 分别降低约 90% 和 99%。

正式性能测试后续需要：

```text
Warm-up
Repeated Runs
Mean Latency
P95 Latency
```

---

## 25. Why Coreset Is Needed

Random Sampling 的问题：

```text
常见正常模式：
可能随机保留很多

稀有正常模式：
可能一个都没保留
```

当某种正常视觉模式被随机删除后：

```text
Normal Query Patch
↓
找不到正确正常邻居
↓
只能匹配较差 Normal Patch
↓
Anomaly Score 被错误抬高
```

Coreset 的目标不是简单：

```text
减少 Memory Size
```

而是：

> 使用更少的 Patch，同时尽可能覆盖完整 Normal Feature Space。

因此：

```text
Random Sampling:
减少数量

Coreset Sampling:
减少数量 + 尽量保留代表性
```

EXP013 为学习 Coreset 提供了直接实验动机。

---

## 26. Prediction vs Result

### Prediction 1

预测：

```text
score_full <= score_10% <= score_1%
```

实际：

```text
Full <= 10%:
True

10% <= 1%:
True
```

结果：

```text
PASS
```

---

### Prediction 2

预测：

```text
10% Memory 约减少 90% workload
```

实际：

```text
Full:
128,463,104

10%:
12,845,840
```

结果：

```text
PASS
```

---

### Prediction 3

预测：

```text
1% Memory 约减少 99% workload
```

实际：

```text
1,284,192
```

结果：

```text
PASS
```

---

### Prediction 4

预测：

```text
10% 仍能较好保留 Full Score Map
```

实际：

```text
Correlation:
0.981391
```

结果：

```text
PASS
```

---

### Prediction 5

预测：

```text
1% 开始明显损失正常模式覆盖
```

实际：

```text
Mean:
0.778466 → 1.204370

Correlation:
0.893151

MAE:
0.425904
```

结果：

```text
PASS
```

---

## 27. Conclusion

EXP013 成功验证了 Memory Bank Size 对 Patch-level anomaly detection 的影响。

完整关系：

```text
Memory Bank Larger
↓
Normal Feature Coverage Better
↓
Nearest Normal Neighbor Closer
↓
Scores Lower / More Stable
↓
Compute and Memory Higher
```

反之：

```text
Memory Bank Smaller
↓
Compute Lower
↓
Normal Coverage Worse
↓
Normal Patch Scores Rise
↓
Potential False Positive Risk Higher
```

本实验得到：

```text
10% Random Memory:

90% workload reduction
Correlation = 0.981391
较好保留 Full Score Map


1% Random Memory:

99% workload reduction
Correlation = 0.893151
Mean Score 明显上升
正常模式覆盖明显损失
```

因此：

> Normal Memory Bank 中存在大量冗余，但过度随机压缩会损伤正常 Feature Space 的覆盖能力。

这直接说明了 Coreset Selection 的必要性。

---

## 28. Limitation

当前 Ablation 只基于：

```text
broken_small/000.png
```

单个测试样本。

因此不能声明：

```text
10% Random Memory
是 bottle 数据集最优配置
```

当前只能得出：

> 在该样本上，10% Random Memory 显示出较好的 Score-map Preservation 和显著的 Compute Reduction，而 1% 已经出现明显的 Normal Coverage Loss。

后续需要：

```text
更多 Test Images
Whole Dataset AUROC
Normal vs Defect Score Distribution
Coreset Comparison
```

进一步验证。

---

## 29. Saved Artifact

结果保存：

```text
results/exp013/
memory_bank_size_ablation.pt
```

包含：

```text
random_seed

full_scores
scores_10
scores_1

memory_size_full
memory_size_10
memory_size_1

elapsed_10
elapsed_1

comparison_10
comparison_1
```

---

## 30. New Questions

1. 同样保留 10%，Coreset 是否会比 Random Sampling 更稳定？
2. 如何选择具有代表性的 Normal Patch？
3. Coreset 的目标函数是什么？
4. Coreset 如何覆盖整个 Normal Feature Space？
5. 10% Coreset 是否能降低正常区域 Score Inflation？
6. Random 10% 与 Coreset 10% 的 Correlation、MAE 有何区别？
7. Coreset Reduction 后 Whole Test Set AUROC 是否能保持？
8. Memory Size 与 Latency 应如何正式评估？

---

## 31. Next Step

下一阶段：

```text
Coreset Selection
```

目标：

```text
Full Normal Memory Bank
↓
Representative Patch Selection
↓
Coreset Memory Bank
```

然后进行：

```text
Random 10%
vs
Coreset 10%
```

公平对照。

核心问题：

> 同样只保留 10% Patch，是否可以通过“选择更有代表性的 Patch”，比随机抽样更好地保留 Normal Feature Space？

---

## Status

```text
Nested Subset Construction:
PASS

Mathematical Prediction:
PASS

Memory Size Ablation:
PASS

Score-map Correlation Analysis:
PASS

Compute Workload Analysis:
PASS

Parameter Behavior Understanding:
PASS

EXP013:
PASS

Current Learning Unit:
CONTINUE
```