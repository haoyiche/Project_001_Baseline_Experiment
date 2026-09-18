# EXP015 - Random vs Coreset Mini Evaluation

## 1. Goal

本实验的目标是：

将 EXP014 中的 K-center Coreset 真正接回 Patch-level anomaly detection pipeline，并验证：

> K-center 在 Normal Feature Space 中获得的 Coverage Improvement，是否能够转化为实际异常检测性能提升。

公平比较：

```text
Candidate Reference:
5000 Normal Patches

Random:
100 Normal Patches

K-center Coreset:
100 Normal Patches
```

Evaluation Set：

```text
20 Good Images

20 Broken Small Images

Total:
40 Images
```

核心问题：

1. Random100 是否会因 Normal Coverage Loss 导致正常图片 anomaly score 被抬高？
2. Coreset100 是否能降低正常图片 Score Upper Tail？
3. Coreset100 是否能在只保留 100 个 Patch 时接近 Candidate5000 的 anomaly ranking performance？
4. Feature-space Coverage Improvement 是否真的能够转化成 Detection Improvement？
5. Memory Size 与 Memory Selection Strategy 哪一个更重要？

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

Feature Layer:
layer2

Device:
CPU

Dataset:
MVTec AD

Category:
bottle

Normal Class:
good

Anomaly Class:
broken_small
```

---

## 3. Memory Banks

Full Memory Bank：

```text
[163856,128]
```

但 EXP015 不直接比较 Full Bank。

首先使用 EXP014 保存的：

```text
candidate_indices
```

重建：

```text
Candidate Reference:
[5000,128]
```

然后从 Candidate Pool 中根据 EXP014 保存的 indices 重建：

```text
Random 100:
[100,128]

K-center Coreset 100:
[100,128]
```

结构：

```text
Full Memory 163856
        ↓
Candidate 5000
        ↓
 ┌───────────────┐
Random 100   Coreset 100
```

Random 和 Coreset：

```text
Candidate Pool 相同
Memory Size 相同
Feature Dimension 相同
Query Image 相同
Distance Metric 相同
Image Score Aggregation 相同
```

唯一主要变量：

```text
Memory Selection Strategy
```

---

## 4. Evaluation Set

Good：

```text
good/000.png
...
good/019.png
```

共：

```text
20
```

Broken Small：

```text
broken_small/000.png
...
broken_small/019.png
```

共：

```text
20
```

Total：

```text
40 images
```

这是一个：

```text
Mini Evaluation
```

不是完整 Bottle Test Set。

---

## 5. Patch-Level Anomaly Score

对测试图片中的每个 Query Patch：

\[
q_i
\]

Memory Bank：

\[
M
\]

Patch anomaly score：

\[
s(q_i)
=
\min_{m\in M}
\|q_i-m\|_2
\]

含义：

```text
Query Patch
↓
与所有 Normal Memory Patch 比较
↓
找到最近 Normal Feature
↓
Nearest Neighbor Distance
↓
Patch Anomaly Score
```

---

## 6. Image-Level Anomaly Score

一张图片具有：

```text
784 Patch Scores
```

当前 minimal baseline 使用：

\[
S(x)
=
\max_i s(q_i)
\]

即：

```text
[784] Patch Scores
↓
Maximum
↓
1 Image Anomaly Score
```

该设计来自 EXP011 的观察：

```text
Local Small Defect
可能只影响少量 Patch

Mean Score
容易被大量正常区域稀释

Max Score
更容易保留局部强异常信号
```

注意：

当前仍属于：

```text
Minimal PatchCore-style Baseline
```

并不是原始 PatchCore 完整的 image-score reweighting。

---

## 7. Prediction

### Prediction 1

预计：

```text
broken_small score
整体高于
good score
```

但 Good / Defect Distribution 可能存在 overlap。

---

### Prediction 2

Random100 的 Normal Feature Coverage 较差。

因此部分 Good Image 可能产生：

```text
较高 Max Patch Score
```

Good Score Upper Tail 预计升高。

---

### Prediction 3

EXP014 中 Coreset100 显著改善了：

```text
P95 Coverage
Maximum Coverage
```

因此假设：

> Coreset100 可能降低部分 Good Image 的极端高 anomaly score。

---

### Prediction 4

Coreset Coverage 更好：

```text
不代表 AUROC 数学上一定更高
```

是否真正改善 detection 必须由实际数据验证。

---

### Prediction 5

Candidate5000 拥有更多 Normal Pattern。

但：

```text
Memory Size Larger
```

并不自动保证所有最终指标一定最好。

必须比较：

```text
Good Distribution
Defect Distribution
Ranking
AUROC
```

---

## 8. Core Code

### 8.1 Reconstruct EXP014 Memory Banks

首先重建 Full Memory：

```python
full_memory_bank = build_memory_bank(...)
```

然后：

```python
candidate_bank = full_memory_bank[
    candidate_indices
]
```

得到：

```text
[5000,128]
```

注意：

```text
candidate_indices
```

属于 Full Memory Bank 坐标系。

---

Random / Coreset indices 则属于：

```text
Candidate Pool 坐标系
```

因此：

```python
random_bank = candidate_bank[
    random_indices
]
```

以及：

```python
coreset_bank = candidate_bank[
    coreset_indices
]
```

不能直接写：

```python
full_memory_bank[random_indices]
```

否则会访问错误的 Patch。

这是一个典型：

```text
Index Coordinate System
```

问题。

---

## 9. Extract Query Feature Once

每张测试图只进行一次：

```python
query_patches = extract_image_patches(...)
```

得到：

```text
[784,128]
```

然后同一份 Query Feature 分别与：

```text
Candidate5000
Random100
Coreset100
```

比较。

这样可以确保实验只改变：

```text
Memory Selection Strategy
```

而不会引入 Feature Extractor 差异。

---

## 10. Exact Nearest Neighbor

代码：

```python
distances = torch.cdist(
    query_patches,
    memory_bank,
    p=2,
)
```

Candidate5000 时：

```text
Query:
[784,128]

Memory:
[5000,128]
```

Distance Matrix：

```text
[784,5000]
```

然后：

```python
patch_scores = (
    distances.min(dim=1).values
)
```

得到：

```text
[784]
```

最后：

```python
image_score = (
    patch_scores.max().item()
)
```

得到：

```text
1 image score
```

---

## 11. Why No Chunking Is Needed

Candidate5000 的完整距离矩阵：

\[
784\times5000
=
3,920,000
\]

float32 大约：

```text
15 MiB
```

当前 CPU 环境可以直接计算。

而 Full Memory：

\[
784\times163856
=
128,463,104
\]

约：

```text
490 MiB
```

因此 EXP011 才需要 Chunked Search。

工程原则：

> 是否 Chunk，应根据实际 Memory Scale 决定，而不是机械使用同一种实现。

---

## 12. Candidate 5000 Result

### Good

```text
Minimum:
2.594655

Mean:
3.320070

P95:
4.214130

Maximum:
4.331443
```

### Broken Small

```text
Minimum:
3.679756

Mean:
4.353979

Maximum:
4.956395
```

### Separation Margin

定义：

\[
Margin
=
Defect_{min}
-
Good_{max}
\]

结果：

```text
3.679756
-
4.331443
=
-0.651687
```

因此：

```text
Good / Defect distribution overlap
```

### AUROC

```text
0.957500
```

---

## 13. Random 100 Result

### Good

```text
Minimum:
4.653347

Mean:
4.940541

P95:
5.095362

Maximum:
5.512643
```

### Broken Small

```text
Minimum:
4.780574

Mean:
5.081893

Maximum:
6.130684
```

### Separation Margin

```text
-0.732069
```

### AUROC

```text
0.560000
```

---

## 14. K-Center Coreset 100 Result

### Good

```text
Minimum:
3.433049

Mean:
3.777570

P95:
4.361700

Maximum:
4.635723
```

### Broken Small

```text
Minimum:
3.993357

Mean:
4.554024

Maximum:
5.086922
```

### Separation Margin

```text
-0.642365
```

### AUROC

```text
0.955000
```

---

## 15. Main Comparison

| Method | Memory Size | Good Mean | Good P95 | Good Max | Defect Mean | Margin | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 5000 | 3.320070 | 4.214130 | 4.331443 | 4.353979 | -0.651687 | 0.957500 |
| Random | 100 | 4.940541 | 5.095362 | 5.512643 | 5.081893 | -0.732069 | 0.560000 |
| Coreset | 100 | 3.777570 | 4.361700 | 4.635723 | 4.554024 | -0.642365 | 0.955000 |

---

## 16. Random vs Coreset

### Good P95

```text
Random:
5.095362

Coreset:
4.361700
```

Coreset 明显降低 Normal Score Upper Tail。

---

### Good Max

```text
Random:
5.512643

Coreset:
4.635723
```

说明最极端的正常高分也明显下降。

---

### Separation Margin

```text
Random:
-0.732069

Coreset:
-0.642365
```

Coreset Margin 略有改善。

但二者均：

```text
< 0
```

因此当前 40 张图仍不能被单一 Threshold 完美分开。

---

### AUROC

```text
Random:
0.560000

Coreset:
0.955000
```

差异：

```text
0.395
```

这是本实验最显著的结果。

---

## 17. Why Random 100 Fails

Random100 相比 Candidate5000：

```text
Good Mean:

3.320070
→
4.940541
```

约增加：

```text
48.8%
```

而 Defect Mean：

```text
4.353979
→
5.081893
```

约增加：

```text
16.7%
```

因此 Random Reduction 的主要问题不是：

```text
Defect Score 不够高
```

而是：

```text
Normal Score 被过度抬高
```

因 Normal Feature Coverage 丢失：

```text
Normal Query Patch
↓
对应正常模式没有被 Random100 保留
↓
找不到合适最近邻
↓
只能匹配更远的 Normal Patch
↓
NN Distance 增大
↓
Normal Anomaly Score 上升
```

最终造成：

```text
Good / Defect Separation
严重下降
```

---

## 18. Mean Class Separation

Candidate：

\[
4.353979-3.320070
=
1.033909
\]

Coreset：

\[
4.554024-3.777570
=
0.776454
\]

Random：

\[
5.081893-4.940541
=
0.141352
\]

可以直观表示：

```text
Candidate:

Good ───────────── Defect


Coreset:

Good ───────── Defect


Random:

Good ─ Defect
```

Random100 几乎将两类 Score Distribution 挤在一起。

---

## 19. Why Coreset 100 Works

EXP014 已经证明：

```text
Random Maximum Coverage:
5.378171

Coreset Maximum Coverage:
3.450252
```

说明 Coreset 对：

```text
Rare / Sparse Normal Pattern
```

的覆盖明显更好。

这在 EXP015 中传导为：

```text
Normal Query
↓
更容易找到合理 Coreset Neighbor
↓
Normal NN Distance 降低
↓
Good Max Score 降低
↓
Good Upper Tail 降低
↓
Good / Defect Ranking 恢复
```

因此：

```text
Random AUROC:
0.5600

Coreset AUROC:
0.9550
```

---

## 20. Coverage → Detection Evidence Chain

当前 EXP014 + EXP015 形成完整证据链：

```text
K-center Selection
↓
改善 Normal Worst-case Coverage
↓
保留更多 Rare Normal Pattern
↓
Normal Query 更容易找到相似 Normal Neighbor
↓
Normal Patch NN Distance 降低
↓
Good Image Extreme Score 降低
↓
Good / Defect Ranking 改善
↓
AUROC 恢复
```

这是目前 Coreset Learning Unit 的核心结论。

---

## 21. Coreset 100 vs Candidate 5000

Candidate：

```text
AUROC:
0.9575
```

Coreset：

```text
AUROC:
0.9550
```

差异：

```text
0.0025
```

Memory Size：

```text
5000
↓
100
```

即：

```text
50× reduction
```

在当前 40-image mini evaluation 中：

> Coreset100 使用 Candidate5000 仅 2% 的 Memory，却几乎保留了相同的 image-level ranking performance。

这是非常重要的压缩结果。

---

## 22. AUROC Pair Interpretation

Evaluation Set：

```text
20 Good
20 Defect
```

因此总共有：

\[
20\times20
=
400
\]

组：

```text
Good vs Defect
```

配对。

在当前无 tie 情况下，可以近似理解：

### Candidate 5000

\[
0.9575\times400
=
383
\]

即：

```text
383 / 400
```

Good-Defect Pair 排序正确。

---

### Coreset 100

\[
0.9550\times400
=
382
\]

即：

```text
382 / 400
```

排序正确。

---

### Random 100

\[
0.56\times400
=
224
\]

即：

```text
224 / 400
```

排序正确。

因此 Candidate 与 Coreset 在当前 mini set 上只相差约：

```text
1 个 Pair Ranking
```

而 Random 则损失严重。

---

## 23. Why Negative Margin Does Not Contradict High AUROC

Candidate：

```text
Good Max:
4.331443

Defect Min:
3.679756
```

所以：

```text
Margin:
-0.651687
```

说明至少存在：

```text
某个 Good
>
某个 Defect
```

因此不存在一个 Threshold 完美分开所有样本。

但 AUROC：

```text
0.9575
```

仍然很高。

原因：

```text
Margin:
只看两个极端点

AUROC:
考虑所有 Good-Defect Pair Ranking
```

因此：

```text
Negative Margin
+
High AUROC
```

完全可以同时存在。

---

## 24. Candidate vs Coreset Metric Difference

Candidate：

```text
Margin:
-0.651687

AUROC:
0.9575
```

Coreset：

```text
Margin:
-0.642365

AUROC:
0.9550
```

按 Margin：

```text
Coreset 略好
```

按 AUROC：

```text
Candidate 略好
```

这是因为：

```text
Margin
只关注最极端的 Good / Defect

AUROC
关注全部 Pair Ranking
```

因此不同 Metric 不一定产生相同排序。

---

## 25. Why High Defect Score Alone Is Not Enough

Random：

```text
Defect Mean:
5.081893
```

看起来非常高。

但是：

```text
Good Mean:
4.940541
```

也非常高。

异常检测真正需要的是：

\[
Score_{defect}
\gg
Score_{good}
\]

而不是：

```text
Defect Score absolute value very large
```

例如：

```text
Good = 100
Defect = 101
```

并不一定是好的 anomaly detector。

而：

```text
Good = 1
Defect = 5
```

可能具有更好的 separation。

因此必须分析：

```text
Distribution
Ranking
AUROC
Threshold
```

而不是只看 Defect Max。

---

## 26. Important Samples

### High-Scoring Normal Candidate

```text
good/010.png
```

Scores：

```text
Candidate:
4.331443

Random:
5.512643

Coreset:
4.635723
```

它在多个 Memory Strategy 下均属于高分 Normal Sample。

因此可作为后续：

```text
High-scoring Normal
FP-risk Candidate
```

进行 Failure Analysis。

---

### Low-Scoring Defect Candidate

```text
broken_small/015.png
```

Scores：

```text
Candidate:
3.679756

Random:
4.832803

Coreset:
3.993357
```

它是当前较难检测的 broken_small 样本。

因此可作为：

```text
Low-scoring Defect
Missed-risk Candidate
```

进行后续分析。

当前没有 Threshold，因此尚不能正式称：

```text
FP
FN
```

---

## 27. Candidate Score Monotonicity

因为：

\[
Random100
\subset
Candidate5000
\]

以及：

\[
Coreset100
\subset
Candidate5000
\]

对任意 Query Patch：

\[
s(q,Candidate)
\le
s(q,Random)
\]

同时：

\[
s(q,Candidate)
\le
s(q,Coreset)
\]

因为更大的 Candidate Bank 具有更多最近邻候选。

从当前 40-image 输出可以看到：

```text
Candidate Image Score
始终 <=
Random / Coreset Score
```

符合最近邻理论。

---

## 28. Interesting Exact Matches

部分 Defect Sample：

```text
broken_small/004

Candidate:
4.776104

Coreset:
4.776104
```

以及：

```text
broken_small/005

Candidate:
4.900589

Coreset:
4.900589
```

说明决定这些图片最大 anomaly score 的关键 Query Patch：

```text
在 Candidate5000 中的关键最近邻
被 Coreset100 成功保留
```

因此删掉另外大量 Normal Patch 后：

```text
关键异常响应没有改变
```

这体现了 Representative Selection 的价值。

---

## 29. Prediction vs Result

### Prediction 1

预测：

```text
Broken Small 整体高于 Good，
但存在 overlap。
```

实际：

```text
三种方法 Margin 均 < 0
但 Defect Mean 均 > Good Mean
```

Verdict：

```text
PASS
```

---

### Prediction 2

预测：

```text
Random100 可能使 Good Extreme Score 升高。
```

实际：

```text
Candidate Good P95:
4.214130

Random Good P95:
5.095362
```

以及：

```text
Candidate Good Max:
4.331443

Random Good Max:
5.512643
```

Verdict：

```text
PASS
```

---

### Prediction 3

预测：

```text
Coreset100 可能降低 Good Upper Tail。
```

实际：

```text
Random Good P95:
5.095362

Coreset Good P95:
4.361700
```

以及：

```text
Random Good Max:
5.512643

Coreset Good Max:
4.635723
```

Verdict：

```text
PASS
```

---

### Prediction 4

预测：

```text
Coverage Improvement 是否提高 AUROC
必须由实际实验验证。
```

实际：

```text
Random:
0.560000

Coreset:
0.955000
```

当前 mini set 强烈支持：

```text
Coverage Improvement
成功转化为 Detection Improvement
```

---

## 30. Engineering Observation

当前代码每次 EXP 都会重新：

```text
build full memory bank
```

随着实验增加，这会产生重复计算。

后续建议缓存：

```text
memory_bank.pt
```

同时保存 metadata：

```text
dataset
category
backbone
weights
feature_layer
input_size
feature_dimension
number_of_patches
preprocessing
```

否则未来修改模型配置后，很容易错误复用旧 Feature。

---

## 31. Limitation

当前结果非常有意义，但只能覆盖：

```text
20 Good
20 Broken Small
```

还没有评估：

```text
broken_large
contamination
全部 good
整个 bottle test set
```

也尚未与：

```text
Full Memory 163856
```

进行同批全量评价。

因此不能直接声明：

```text
Coreset100
已经等价于 Full PatchCore Memory
```

当前准确结论是：

> 在当前 40-image mini evaluation 上，K-center Coreset100 使用 Candidate5000 仅 2% 的 Memory，却基本保持了 Candidate5000 的 image-level AUROC，而 Random100 在相同容量下出现严重 Normal Coverage Loss 和检测性能下降。

---

## 32. Conclusion

EXP015 将 EXP014 的 Feature Coverage 分析真正接入 anomaly detection task。

结果：

```text
Candidate5000:
AUROC = 0.9575

Random100:
AUROC = 0.5600

Coreset100:
AUROC = 0.9550
```

Random100 的主要失败机制：

```text
Random Reduction
↓
Rare Normal Pattern 丢失
↓
Normal Query 找不到合适最近邻
↓
Good Score 被大幅抬高
↓
Good / Defect Separation 消失
↓
AUROC 显著下降
```

K-center Coreset：

```text
改善 Worst-case Normal Coverage
↓
保留更多 Feature Space 边缘和稀有正常模式
↓
降低 Good Score Upper Tail
↓
恢复 Good / Defect Ranking
↓
AUROC 接近 Candidate5000
```

因此本实验提供了强证据支持：

> Coreset 的价值不只是缩小 Memory Bank，而是在压缩 Memory 的同时尽量保留 Normal Feature Space Coverage，从而保护 anomaly detection performance。

---

## 33. New Questions

1. Coreset100 在完整 Bottle Test Set 上还能保持性能吗？
2. `broken_large` 和 `contamination` 是否具有相同规律？
3. Candidate5000 与 Full163856 的 AUROC 差异是多少？
4. Coreset Ratio 从 2% 增大到 5%、10% 会怎样？
5. Max Image Score 是否是最佳 aggregation？
6. Top-K Mean 是否会比 Max 更稳定？
7. Global Feature Baseline 与 Patch-level Baseline 在同一 Test Set 上谁更适合局部缺陷？
8. 如何选择正式 Threshold？
9. `good/010` 为什么持续高分？
10. `broken_small/015` 为什么持续低分？

---

## 34. Next Step

后续进入：

```text
Full Bottle Test Evaluation
```

需要覆盖：

```text
good
broken_large
broken_small
contamination
```

并开始正式比较：

```text
Global Feature Baseline
vs
Patch-level Baseline
```

同时记录：

```text
Image-level AUROC
ROC / PR
Good / Defect Score Distribution
Failure Cases
```

---

## Status

```text
EXP014 Bank Reconstruction:
PASS

40-image Mini Evaluation:
PASS

Random100 Evaluation:
PASS

Coreset100 Evaluation:
PASS

Good Upper-tail Analysis:
PASS

AUROC Analysis:
PASS

Coverage → Detection Verification:
PASS

EXP015:
PASS

Patch-level Anomaly Detection:
CONTINUE

Git:
暂不提交
```