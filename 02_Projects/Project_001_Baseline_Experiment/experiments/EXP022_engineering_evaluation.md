# EXP022：工程性能评估与端到端瓶颈分析

## 1. 实验目标

本实验的目标不是继续单纯追求异常检测精度，而是评估已有方法在真实工程部署条件下的性能表现。

本实验比较两种已经完成准确率验证的方法：

- **Global**
  - ResNet18 完整 backbone
  - 使用 layer4 之后的 512 维 global feature
  - 使用正常训练样本的 feature mean 作为 normal center
  - 测试图像通过与 normal center 的 L2 距离得到 anomaly score

- **Patch-layer3**
  - ResNet18 截断到 layer3
  - 每张图得到 `14 × 14 = 196` 个 patch feature
  - 每个 patch feature 为 256 维
  - 使用 Candidate5000 + K-center Coreset100
  - 每个 query patch 与 100 个 normal prototype 计算最近邻距离
  - 使用 MAX aggregation 得到 image-level anomaly score

本实验重点回答：

1. 两种方法谁的在线推理 latency 更低？
2. Mean latency 和 P95 latency 分别如何？
3. 单图顺序推理 throughput 如何？
4. 两种方法的 reference representation memory 有多大？
5. Peak GPU memory 如何？
6. Offline reference preparation 和 model startup 如何？
7. Model-only latency 和真实 End-to-End latency 有多大差异？
8. 当前系统真正的性能瓶颈在哪里？

---

## 2. 前置实验与固定配置

本实验建立在以下实验结果之上：

- EXP018：Leakage-safe reference / validation split
- EXP019：Coreset stochastic stability
- EXP020：Image score aggregation ablation
- EXP021：Feature layer ablation

EXP022 使用与 EXP018 / EXP021 一致的数据划分和模型逻辑。

固定配置：

| 项目 | 配置 |
|---|---|
| Dataset | MVTec AD Bottle |
| Reference normal | 167 images |
| Test | 83 images |
| Backbone | ImageNet pretrained ResNet18 |
| Global representation | 512-D |
| Patch layer | layer3 |
| Patch feature map | `[256,14,14]` |
| Query patches/image | 196 |
| Patch feature dim | 256 |
| Candidate size | 5000 |
| Coreset size | 100 |
| Patch seed | 42 |
| Aggregation | MAX |
| Online batch size | 1 |
| Warm-up | 20 iterations |
| Benchmark repeats | 5 |
| Offline batch size | 16 |

---

## 3. CUDA 环境修复

### 3.1 初始状态

硬件和 NVIDIA 系统环境正常：

```text
GPU:
NVIDIA GeForce RTX 2060 with Max-Q Design

Driver:
552.22

nvidia-smi CUDA:
12.4

CUDA Toolkit / nvcc:
12.4
```

但 Python 环境最初为：

```text
torch: 2.14.0+cpu
torchvision: 0.29.0+cpu
torch.version.cuda: None
torch.cuda.is_available(): False
```

因此问题不在 GPU 或 CUDA Toolkit，而在于当前 Conda 环境安装的是 **CPU-only PyTorch build**。

随后更换为：

```text
torch: 2.14.0+cu126
torchvision: 0.29.0+cu126
torch CUDA build: 12.6
```

验证：

```text
torch.cuda.is_available(): True
device count: 1
GPU: NVIDIA GeForce RTX 2060 with Max-Q Design
```

实际 CUDA tensor matrix multiplication 测试：

```text
device: cuda:0
allocated: 40.125 MiB
```

并且：

```text
python -m pip check

No broken requirements found.
```

因此 CUDA 环境修复：

**PASS**

---

## 4. CPU → CUDA 功能回归验证

在正式进行性能测试之前，使用 EXP021：

```text
layer2
seed42
Candidate5000
Coreset100
MAX
```

进行 CPU → CUDA functional regression check。

结果：

```text
Expected AUROC : 0.984921
CUDA AUROC     : 0.984921
Absolute delta : 0.00000037
AP             : 0.995441
```

误差：

\[
3.7\times10^{-7}
\]

远小于：

\[
10^{-6}
\]

说明更换 CUDA PyTorch 后，没有改变已有 anomaly detection pipeline 的实质行为。

结论：

**CUDA regression PASS**

---

## 5. 实验前预测

### 5.1 Online latency

初始预测：

Global 的 anomaly scoring 很轻：

\[
512D\ feature
\rightarrow
normal\ center\ L2
\]

Patch-layer3 需要：

\[
196\times100
\]

次 patch/prototype distance 计算。

但 Patch 在 layer3 就停止，而 Global 还要继续执行 layer4。

因此在真正 benchmark 之前，无法仅凭直觉确定哪种方法更快。

### 5.2 Representation memory

Global 保存一个 512 维 float32 center：

\[
512\times4=2048\ bytes
\]

约：

\[
2KiB
\]

Patch 保存：

\[
100\times256
\]

个 float32：

\[
100\times256\times4
=
102400\ bytes
\]

约：

\[
100KiB
\]

因此 Patch reference representation memory 约为 Global 的：

\[
50\times
\]

### 5.3 Mean 与 P95

Mean latency：

> 描述平均一次请求的耗时。

P95 latency：

> 大约 95% 的请求 latency 不超过该值。

P95 在典型右偏 latency distribution 中通常高于 mean，但两者不存在数学上的固定大小关系。

### 5.4 Latency 与 Throughput

在 batch size = 1 且顺序执行的近似条件下：

\[
Throughput
\approx
\frac{1000}{Latency(ms)}
\]

例如：

\[
20ms\rightarrow50\ images/s
\]

\[
50ms\rightarrow20\ images/s
\]

---

## 6. Benchmark 代码设计

主要脚本：

```text
src/experiments/engineering_benchmark.py
```

主要输出：

```text
results/exp022/engineering_benchmark_summary.csv
results/exp022/engineering_latency_samples.csv
```

后续 stage profiler：

```text
src/experiments/engineering_stage_profile.py
```

输出：

```text
results/exp022/engineering_stage_profile_summary.csv
results/exp022/engineering_stage_profile_samples.csv
```

---

## 7. 代码关键逻辑

### 7.1 Patch layer3 backbone

完整 ResNet18：

```text
stem
→ layer1
→ layer2
→ layer3
→ layer4
→ avgpool
→ fc
```

Patch-layer3 截断为：

```text
stem
→ layer1
→ layer2
→ layer3
```

因此输入：

\[
[1,3,224,224]
\]

输出：

\[
[1,256,14,14]
\]

### 7.2 Patch feature reshape

feature map：

\[
[1,256,14,14]
\]

经过：

```python
permute(0, 2, 3, 1)
```

变为：

\[
[1,14,14,256]
\]

随后 reshape：

\[
[196,256]
\]

其中：

- 196：`14 × 14` 个空间 patch
- 256：每个 patch 的 feature dimension

### 7.3 Patch nearest-neighbor scoring

Query：

\[
[196,256]
\]

Coreset：

\[
[100,256]
\]

执行：

```python
torch.cdist(patches, coreset)
```

得到：

\[
[196,100]
\]

含义：

> 每个测试 patch 与每个 normal prototype 的 L2 distance。

随后：

```python
distances.min(dim=1)
```

得到：

\[
[196]
\]

每个 query patch 找距离最近的正常 prototype。

再执行：

```python
patch_scores.max()
```

得到一个 image-level anomaly score。

因此：

\[
[196,256]
\rightarrow
[196,100]
\rightarrow
[196]
\rightarrow
1
\]

---

## 8. CUDA latency 测量原则

CUDA 默认异步执行。

因此：

```python
score = model(x)
```

返回时并不一定代表 GPU 已经完成运算。

为了正确测量 GPU latency，需要：

```python
torch.cuda.synchronize()

start = time.perf_counter()

score = model(x)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start
```

计时前同步：

> 保证之前的 GPU 工作已经完成。

计时后同步：

> 等待当前 inference 真正完成。

否则测到的可能只是 CPU 提交 CUDA kernel 所需要的时间。

---

## 9. 第一阶段：Model-Only Benchmark

Model-only measurement boundary：

```text
CPU Tensor
→ GPU
→ Backbone
→ Anomaly Scoring
```

不包含：

```text
PNG loading
PIL decode
resize / crop
normalization
```

---

## 10. Model-Only 实验结果

### 10.1 Global

```text
AUROC:     0.976984
AP:        0.991601

Mean latency:   7.940 ms
P95 latency:    8.286 ms
Throughput:     125.945 images/s

Startup:        215.801 ms
Offline prep:   1812.762 ms

Representation: 2.00 KiB
Peak allocated VRAM: 49.39 MiB
```

Global AUROC 与 EXP018：

```text
EXP018: 0.976984
EXP022: 0.976984
delta:  0.00000013
```

说明 benchmark 没有改变算法行为。

### 10.2 Patch-layer3

```text
AUROC:     1.000000
AP:        1.000000

Mean latency:   4.533 ms
P95 latency:    4.954 ms
Throughput:     220.600 images/s

Startup:        186.726 ms
Offline prep:   807.293 ms

Representation: 100.00 KiB
Peak allocated VRAM: 26.50 MiB
```

Patch AUROC 与 EXP021 seed42：

```text
EXP021: 1.000000
EXP022: 1.000000
delta:  0
```

同样成功复现已有结果。

---

## 11. Model-Only 对比

| Metric | Global | Patch-layer3 |
|---|---:|---:|
| AUROC | 0.976984 | 1.000000 |
| AP | 0.991601 | 1.000000 |
| Mean latency | 7.940 ms | 4.533 ms |
| P95 latency | 8.286 ms | 4.954 ms |
| Throughput | 125.945 img/s | 220.600 img/s |
| Representation | 2 KiB | 100 KiB |
| Peak allocated VRAM | 49.39 MiB | 26.50 MiB |

Patch model-only latency 相对 Global 降低：

\[
1-\frac{4.533}{7.940}
\approx42.9\%
\]

Global / Patch latency ratio：

\[
\frac{7.940}{4.533}
\approx1.75
\]

即当前配置下 Patch model-only inference 约快：

\[
1.75\times
\]

---

## 12. 为什么 Patch 反而更快

虽然 Patch 需要计算：

\[
196\times100=19600
\]

组 patch/prototype distance，但 Patch backbone 只执行到 layer3。

Global 需要继续执行：

```text
layer4
avgpool
```

因此：

```text
Global:
backbone 更深
scoring 很轻

Patch:
backbone 更浅
scoring 更重
```

实际结果说明，在当前 RTX 2060 Max-Q 和当前 memory size 下：

> 省掉 layer4 得到的计算收益，大于 Patch NN scoring 引入的额外开销。

---

## 13. Representation Memory 与 Peak VRAM

Patch：

```text
representation = 100 KiB
```

Global：

```text
representation = 2 KiB
```

因此 Patch 大约使用：

\[
50\times
\]

更多 reference representation memory。

但 Peak allocated VRAM：

```text
Global: 49.39 MiB
Patch:  26.50 MiB
```

并不矛盾。

原因是：

> Representation memory 与 total peak inference VRAM 是不同概念。

Peak VRAM 还受到 forward 中 intermediate activation memory 的影响。

Patch 只运行到 layer3，因此虽然 reference representation 更大，但 activation memory 更少。

所以：

```text
Reference representation:
Patch > Global

Peak inference VRAM:
Patch < Global
```

两者不存在必然正相关关系。

---

## 14. 主动修改 1：End-to-End Benchmark

### 14.1 修改目的

Model-only latency 并不代表真实工业视觉系统 latency。

因此增加 End-to-End benchmark。

测量边界改为：

```text
Disk PNG
→ PIL
→ RGB conversion
→ Preprocess
→ CPU Tensor
→ GPU
→ Backbone
→ Anomaly Score
```

### 14.2 修改前预测

已有：

```text
Global model-only = 7.940 ms
Patch model-only  = 4.533 ms

Shared preprocessing ≈ 19.3 ms
```

预测：

1. 两种方法 E2E latency 都会显著增加。
2. Patch 仍有可能保持更低 latency。
3. Patch 相对 Global 的速度优势会明显缩小，因为两者共享相同的输入 pipeline 开销。

---

## 15. End-to-End 实验结果

Shared preprocessing：

```text
Mean = 19.301 ms
P95  = 21.845 ms
```

Global：

```text
E2E mean latency: 29.810 ms
E2E P95 latency:  35.067 ms
E2E throughput:   33.546 images/s
```

Patch：

```text
E2E mean latency: 28.960 ms
E2E P95 latency:  33.549 ms
E2E throughput:   34.531 images/s
```

---

## 16. Model-Only 与 E2E 对比

| Metric | Global | Patch |
|---|---:|---:|
| Model mean | 7.940 ms | 4.533 ms |
| Model P95 | 8.286 ms | 4.954 ms |
| Model throughput | 125.945 img/s | 220.600 img/s |
| E2E mean | 29.810 ms | 28.960 ms |
| E2E P95 | 35.067 ms | 33.549 ms |
| E2E throughput | 33.546 img/s | 34.531 img/s |

Model-only：

\[
42.9\%
\]

latency reduction。

End-to-End：

\[
1-\frac{28.960}{29.810}
\approx2.85\%
\]

latency reduction。

因此：

> Patch 在模型层约 42.9% 的 latency 优势，到完整输入 pipeline 中只剩约 2.85%。

预测得到验证。

---

## 17. 初步工程结论

模型变快，并不意味着整个系统同比例变快。

虽然 Patch 将 model-side latency：

\[
7.940ms
\rightarrow
4.533ms
\]

但 E2E：

\[
29.810ms
\rightarrow
28.960ms
\]

变化很小。

说明当前系统还有大量开销发生在模型之外。

因此不能仅通过 model inference benchmark 判断完整工业视觉 pipeline 的性能。

---

## 18. 主动修改 2：Stage Profiling

### 18.1 目的

将 E2E latency 分解为：

```text
Stage 1: PNG loading / decode / RGB
Stage 2: torchvision preprocessing
Stage 3: CPU → GPU transfer
Stage 4: GPU backbone + anomaly scoring
```

目标：

> 找出真正的系统瓶颈。

### 18.2 修改前预测

预测：

> CPU → GPU transfer 是 E2E pipeline 最大耗时阶段。

理由：

> inference 前必须把 input tensor 从 CPU RAM 传输到 GPU VRAM。

---

## 19. Stage Profiling 结果

### 19.1 Global

| Stage | Mean | P95 | Share |
|---|---:|---:|---:|
| PNG decode + RGB | 15.713 ms | 17.392 ms | 50.75% |
| Preprocess | 4.181 ms | 4.663 ms | 13.51% |
| CPU → GPU | 0.490 ms | 0.600 ms | 1.58% |
| Model + scoring | 10.575 ms | 13.390 ms | 34.16% |

Mean stage sum：

\[
30.960ms
\]

### 19.2 Patch-layer3

| Stage | Mean | P95 | Share |
|---|---:|---:|---:|
| PNG decode + RGB | 15.886 ms | 17.458 ms | 57.61% |
| Preprocess | 4.166 ms | 4.672 ms | 15.11% |
| CPU → GPU | 0.497 ms | 0.602 ms | 1.80% |
| Model + scoring | 7.028 ms | 10.408 ms | 25.48% |

Mean stage sum：

\[
27.577ms
\]

---

## 20. Prediction Verification

预测：

> H2D 是最大瓶颈。

结果：

**FAILED**

实际：

```text
H2D share:
Global ≈ 1.58%
Patch  ≈ 1.80%
```

最大的单一阶段是：

```text
PNG loading / decode / RGB
```

Global：

\[
50.75\%
\]

Patch：

\[
57.61\%
\]

---

## 21. 输入 Pipeline 占比

Decode + preprocess：

Global：

\[
50.75\%+13.51\%
=
64.26\%
\]

Patch：

\[
57.61\%+15.11\%
=
72.72\%
\]

因此，模型被进一步加速后：

> 输入 pipeline 反而成为更明显的系统瓶颈。

这说明工程优化必须基于 profiling，而不能仅凭直觉。

---

## 22. 为什么 H2D 并不慢

输入 Tensor：

\[
[1,3,224,224]
\]

元素数量：

\[
3\times224\times224
=
150528
\]

float32：

\[
150528\times4
=
602112 bytes
\]

约：

\[
0.6MB
\]

因此单张图片的 H2D transfer 数据量并不大。

实际测量：

```text
≈ 0.5 ms
```

与数据规模一致。

---

## 23. 为什么 PNG decode 较慢

PNG 不是一个已经展开好的 tensor。

需要执行：

```text
磁盘读取
→ 压缩格式解析
→ PNG decode
→ 像素恢复
→ RGB conversion
```

因此该阶段主要由 CPU / IO / image codec path 完成。

当前实际耗时：

```text
≈ 15.8 ms
```

明显高于 H2D transfer。

---

## 24. Stage Profiling 为什么不能代替 E2E Benchmark

真实 E2E：

```text
start

decode
preprocess
H2D
model

CUDA synchronize
end
```

Stage profiling 为了分别测量 H2D 和 GPU computation，需要额外插入：

```python
torch.cuda.synchronize()
```

例如：

```text
H2D
→ synchronize
→ model
→ synchronize
```

这会改变原来连续的 execution process。

因此：

> Stage profiling 存在 measurement disturbance。

所以：

- **E2E benchmark** 用来回答：完整请求实际需要多久。
- **Stage profiler** 用来回答：时间主要花在哪。
- Stage sum 不应该被当作真实 E2E latency 的替代值。

---

## 25. Amdahl's Law

本实验直接体现了 Amdahl's Law。

假设 Patch 当前大致为：

```text
Decode       ≈ 16 ms
Preprocess   ≈ 4 ms
H2D          ≈ 0.5 ms
Model        ≈ 7 ms
```

总计：

\[
27.5ms
\]

即使更强 GPU 将 model 从：

\[
7ms
\]

优化至：

\[
2ms
\]

系统仍需要：

\[
16+4+0.5+2
=
22.5ms
\]

因此 model 本身虽然：

\[
7/2=3.5\times
\]

加速，但系统整体只有：

\[
27.5/22.5
\approx1.22\times
\]

即约 22% 的整体提升。

原因是：

> Decode、preprocess 和 transfer 没有随着 model 一起加速。

---

## 26. 本实验得到的核心工程认识

1. **Model latency 与系统 E2E latency 是不同指标。**
2. **模型更快，不代表整套系统同比例变快。**
3. **应该先 profiling，再决定优化目标。**
4. 本实验最初预测 H2D 是最大瓶颈，但实验否定了该预测。
5. 当前 pipeline 最大单项开销是 PNG loading / decode / RGB conversion。
6. Decode + preprocess 已占 Patch stage time 的约 72.7%。
7. Representation memory 与 peak inference VRAM 是不同概念。
8. P95 比 mean 更适合观察 tail latency。
9. CUDA benchmark 必须考虑异步执行和 synchronization。
10. Stage profiler 是诊断工具，不能替代真实 E2E benchmark。
11. 进一步加速 GPU model 时，会受到非模型部分的 Amdahl's Law 限制。
12. “模型推理只需要 3 ms”并不等于真实工业视觉系统 E2E 只需要 3 ms。

---

## 27. 当前工程优化优先级

根据 stage profiling，目前不应该优先继续压缩 H2D。

H2D 仅占：

```text
约 1.6%～1.8%
```

当前更值得调查的是：

```text
1. PNG / image decode
2. CPU preprocessing
3. input pipeline architecture
4. model inference
5. H2D
```

这不代表应该立刻开始优化 decode。

正确流程仍然是：

> 测量 → 定位瓶颈 → 提出修改 → 预测 → 验证。

---

## 28. 实验局限性

本实验结果目前只适用于当前实验条件：

```text
GPU:
RTX 2060 Max-Q

Dataset:
MVTec AD Bottle

Input:
224 × 224

Batch:
1

Patch:
layer3
Candidate5000
Coreset100
seed42
MAX
```

还存在以下限制：

- 只测试了 Bottle 类别。
- 只测试了一张 GPU。
- 当前主要是 batch size = 1 sequential inference。
- Patch 固定使用 seed42。
- Global 和 Patch 没有采用随机交错顺序进行 latency benchmark。
- Startup measurement 存在执行顺序和 cache 状态影响。
- E2E benchmark 存在系统状态和测量顺序影响。
- Stage profiler 增加了 synchronization，因此会扰动原始 pipeline。
- 当前尚未测试 pinned memory、async transfer、batch inference 或专门 image decoding pipeline。
- 因此不能将当前结果推广为“Patch 在所有硬件、数据集和部署场景下都优于 Global”。

---

## 29. 最终验收

本实验已经能够解释：

- CUDA asynchronous execution 为什么影响 latency 测量。
- `torch.cuda.synchronize()` 为什么必要。
- `[1,256,14,14]` 如何变成 `[196,256]`。
- `[196,256] × [100,256]` 为什么产生 `[196,100]` distance matrix。
- Global 和 Patch anomaly score 如何计算。
- Representation memory 与 peak VRAM 的区别。
- Mean 和 P95 latency 的区别。
- Model-only latency 与 E2E latency 的区别。
- 为什么不能仅凭 GPU benchmark 判断工业视觉系统性能。
- 为什么应该先 profiling 再优化。
- 为什么 H2D 不是当前瓶颈。
- 为什么 stage sum 不能直接替代真实 E2E latency。
- Amdahl's Law 如何限制进一步 GPU 加速的系统收益。

最终通关问题完成情况：

```text
CUDA timing                      PASS
cdist shape                      PASS
representation vs VRAM           PASS
model-only vs E2E                PASS
profiling bottleneck reasoning   PASS
Amdahl's Law                     PASS
stage profiling limitation       PASS
system inference reasoning       PASS
```

---

## 30. EXP022 最终结论

在当前 RTX 2060 Max-Q、batch size 1、MVTec Bottle 条件下：

### Global

```text
AUROC                 0.976984
AP                    0.991601

Model mean            7.940 ms
Model P95             8.286 ms
Model throughput      125.945 img/s

E2E mean              29.810 ms
E2E P95               35.067 ms
E2E throughput        33.546 img/s

Representation        2 KiB
Peak allocated VRAM   49.39 MiB
```

### Patch-layer3

```text
AUROC                 1.000000
AP                    1.000000

Model mean            4.533 ms
Model P95             4.954 ms
Model throughput      220.600 img/s

E2E mean              28.960 ms
E2E P95               33.549 ms
E2E throughput        34.531 img/s

Representation        100 KiB
Peak allocated VRAM   26.50 MiB
```

Patch-layer3 在 model-only benchmark 中明显快于 Global，平均 latency 低约 42.9%。

但进入完整 E2E pipeline 后，latency 优势仅剩约 2.85%。

Stage profiling 进一步发现：

> 当前最大的单阶段开销不是 GPU inference，也不是 CPU→GPU transfer，而是 PNG loading / decoding / RGB conversion。

因此，本实验最重要的工程结论不是简单的：

> “Patch 更快。”

而是：

> **模型级性能提升和系统级性能提升并不等价。只有测量完整 pipeline，并通过 stage profiling 找到真正瓶颈，才能决定下一步优化方向。**

---

## 31. 当前状态

```text
EXP022 理论理解                  PASS
CUDA 环境                       PASS
功能回归                        PASS
最小实验                        PASS
Model-only benchmark            PASS
主动修改 1：E2E                 PASS
主动修改 2：Stage Profile       PASS
主动修改 3：Balanced E2E        PASS
先预测后验证                    PASS
代码关键逻辑理解                PASS
最终通关测试                    PASS
实验笔记                        PASS

Git checkpoint #1              PASS
GitHub push #1                 PASS
Git checkpoint #2              尚未完成
GitHub push #2                 尚未完成

EXP022                          CONTINUE
LEVEL 1                         CONTINUE
```

> 按 AI Lab SOP，在主动修改 3 的结果写入仓库并完成最终 Git commit + GitHub push 之前，不正式宣布 EXP022 完整通关。


---

## 32. 主动修改 3：Interleaved / Balanced-Order E2E Benchmark

### 32.1 实验动机

原始 E2E benchmark 的执行顺序为：

```text
Global 全部测试
→
Patch 全部测试
```

此前观察到 E2E latency 随 repeat 存在一定时间漂移，而原始结果中：

```text
Global E2E mean = 29.810 ms
Patch E2E mean  = 28.960 ms
```

两者仅相差约：

```text
0.850 ms
```

因此执行时间顺序可能成为混杂因素。

为了降低这一影响，本实验对同一张测试图像连续测试两个方法，并交替执行顺序：

```text
Global → Patch
Patch  → Global
```

同时根据 repeat 和 image index 交替 first / second position，使两种方法在测量顺序中的位置基本平衡。

---

### 32.2 实验前预测

1. Patch 的 model-side 计算优势是真实存在的，因此平衡执行顺序后仍可能保持较低 E2E latency。
2. 由于 decode + preprocess 占据大量时间，因此预计 Global 与 Patch 的 E2E 差距仍然较小。
3. 原始约 0.85 ms 的差值可能发生明显变化，甚至改变符号，因此不预测 Patch 一定仍快 0.85 ms。

---

### 32.3 Balanced-Order 结果

#### Global

```text
AUROC:                 0.976984
AP:                    0.991601

Mean latency:          31.794 ms
P95 latency:           35.811 ms
Throughput:            31.453 img/s

Mean when first:       32.001 ms
Mean when second:      31.586 ms
Second - first:        -0.415 ms
```

#### Patch-layer3

```text
AUROC:                 1.000000
AP:                    1.000000

Mean latency:          26.520 ms
P95 latency:           29.490 ms
Throughput:            37.708 img/s

Mean when first:       26.513 ms
Mean when second:      26.526 ms
Second - first:         0.013 ms
```

测量顺序数量：

```text
Global first:          208
Global second:         207

Patch first:           207
Patch second:          208
```

因此 first / second position 基本完全平衡。

---

### 32.4 Paired Comparison

定义：

```text
paired difference
=
Patch latency - Global latency
```

因此：

```text
negative → Patch faster
positive → Global faster
```

结果：

```text
Mean paired difference:      -5.274 ms
Median paired difference:    -5.189 ms
```

共 415 个 Global/Patch 配对样本，其中 406 个配对结果为负，即当前实验中约 97.8% 的配对测量表现为 Patch latency 更低。

Patch 相对 Global 的 mean E2E latency reduction：

\[
1-\frac{26.520}{31.794}
\approx16.6\%
\]

---

### 32.5 Prediction Verification

#### Prediction 1

Patch 平衡顺序后仍可能保持较低 E2E latency。

结果：

**SUPPORTED**

Balanced benchmark 中：

```text
Global = 31.794 ms
Patch  = 26.520 ms
```

Patch 仍然更低。

#### Prediction 2

Global 与 Patch 的 E2E 差距仍然较小。

结果：

**FAILED**

原始顺序 benchmark 差值约：

```text
0.850 ms
```

Balanced benchmark 配对平均差值扩大到：

```text
5.274 ms
```

说明原来的小差距并不稳定。

#### Prediction 3

原始约 0.85 ms 的差值可能明显变化。

结果：

**SUPPORTED**

控制测量顺序后，Global / Patch E2E 差距发生明显变化。

---

### 32.6 工程解释

该实验说明：

> 当两个方法的 latency 差距较小时，benchmark execution order 本身可能成为重要混杂因素。

原始 sequential benchmark：

```text
Global 全部测量
→
Patch 全部测量
```

无法完全区分：

```text
method difference
```

与：

```text
system temporal drift
cache state
CPU scheduling
background system state
```

带来的影响。

Interleaved balanced-order benchmark 通过让两个方法在相邻时间窗口内测量，并平衡 first / second position，降低了该类时间顺序偏差。

因此当前结果支持：

> 在本实验硬件、数据集和 balanced-order protocol 下，Patch-layer3 的 E2E latency 低于 Global。

但不能进一步推广为：

> Patch 在所有硬件和部署系统中固定比 Global 快约 5.27 ms。

该数值仍然只属于当前实验条件。

---

### 32.7 Active Modification 3 Conclusion

本次修改验证了 benchmark protocol 本身也是实验变量。

仅报告一个 latency 数值是不够的，还必须明确：

- measurement boundary
- warm-up
- repeat count
- execution order
- batch size
- hardware
- synchronization strategy

否则性能比较可能混合模型差异和测量方法差异。

Active Modification 3：

**PASS**

---

## 33. EXP022 最终学习总结

### 33.1 本次完成内容

EXP022 已完成：

- CUDA 环境从 CPU-only PyTorch 修复为 CUDA PyTorch。
- CUDA functional regression check。
- Global 与 Patch-layer3 model-only benchmark。
- End-to-End benchmark。
- Stage profiling。
- Balanced-order interleaved E2E benchmark。
- Mean / P95 / throughput / representation memory / peak VRAM 分析。
- CUDA asynchronous timing 与 synchronize 原理理解。
- Amdahl's Law 与系统瓶颈分析。
- Benchmark order confound 的主动验证。
- 三次主动修改及预测 → 验证闭环。

### 33.2 真正掌握的内容

当前能够用自己的话解释：

1. Model-only latency 与 E2E latency 的区别。
2. CUDA 异步执行为什么会导致错误计时。
3. 为什么 GPU benchmark 要使用 `torch.cuda.synchronize()`。
4. `[1,256,14,14] → [196,256] → [196,100] → [196] → scalar` 的 Patch scoring 数据流。
5. Representation memory 与 peak inference VRAM 的区别。
6. Mean latency 与 P95 latency 的区别。
7. 为什么模型加速不等于系统同比例加速。
8. 为什么 profiling 应先于优化。
9. 为什么当前 H2D 不是主要瓶颈。
10. 为什么 stage profiling 不能替代真实 E2E benchmark。
11. Amdahl's Law 如何限制单阶段优化收益。
12. 为什么 benchmark execution order 本身也可能是实验变量。
13. 为什么实验结论必须限定适用条件，不能把当前数值泛化为普遍规律。

### 33.3 仍需保持警惕的内容

以下内容已经理解，但未来实验中仍需要持续主动检查：

- benchmark measurement boundary 是否一致；
- first / second order 是否平衡；
- warm-up 是否充分；
- latency 是否存在系统漂移；
- 是否把单机、单数据集结果过度推广；
- profiler 的同步操作是否改变原始执行流程；
- 模型级指标是否被误当成系统级指标。

### 33.4 EXP022 当前状态

```text
理论理解                    PASS
最小实验                    PASS
主动修改 1                  PASS
主动修改 2                  PASS
主动修改 3                  PASS
先预测后验证                 PASS
笔记                         PASS
通关测试                     PASS
代码可重复运行               PASS

Git checkpoint #1           PASS
GitHub push #1              PASS
Git checkpoint #2           待完成
GitHub push #2              待完成

EXP022                       CONTINUE
LEVEL 1                      CONTINUE
```

### 33.5 下一步

下一步不是继续增加 EXP022 实验，而是完成最终版本控制闭环：

```text
更新 EXP022 Markdown
→ stage 主动修改 3 代码与结果
→ git diff --cached --check
→ git commit
→ git push origin master
→ 验证 local HEAD == origin/master
```

完成后：

```text
EXP022 → LEVEL COMPLETE（实验单元完成）
```

但：

```text
LEVEL 1 → CONTINUE
```

因为 Level 1 还需要继续完成其余视觉基础与 Boss 验收内容。
