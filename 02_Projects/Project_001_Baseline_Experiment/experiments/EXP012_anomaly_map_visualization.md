# EXP012 - Anomaly Map Visualization and Ground Truth Comparison

## 1. Goal

本实验的目标是：

将 EXP011 得到的 `[28,28]` Patch Anomaly Score Map 上采样到模型实际输入空间 `[224,224]`，并与 MVTec AD 的 Ground Truth Mask 对齐和可视化，从而验证高异常响应区域是否真的对应 `broken_small` 的真实缺陷位置。

核心问题：

1. `[28,28]` Patch Score Map 是否能够正确恢复到 `[224,224]` 模型输入坐标系？
2. Ground Truth 是否和模型输入使用相同的几何预处理？
3. EXP011 中的高分 Patch 是否真正落在缺陷区域附近？
4. Top-K 高分 Patch 是否与 Ground Truth 存在空间对应关系？
5. 当前 Patch-level NN 方法是否已经具备定性的局部缺陷定位能力？

本实验重点是：

```text
Qualitative Localization
+
Ground Truth Spatial Verification
```

暂时不进行 Pixel-level AUROC、PRO、IoU、Dice 等完整定量评价。

---

## 2. Environment

```text
Project:
Project_001_Baseline_Experiment

Conda Environment:
ai_base

Framework:
PyTorch

Visualization:
Matplotlib

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

Defect Type:
broken_small

Test Sample:
000.png
```

---

## 3. Input

Test Image：

```text
data/raw/mvtec_anomaly_detection/
bottle/test/broken_small/000.png
```

Ground Truth：

```text
data/raw/mvtec_anomaly_detection/
bottle/ground_truth/broken_small/000_mask.png
```

Normal Patch Memory Bank：

```text
[163856, 128]
```

Query Patches：

```text
[784, 128]
```

Low-resolution Anomaly Map：

```text
[28, 28]
```

Model Input：

```text
[224, 224]
```

---

## 4. Preprocessing Geometry

ResNet18 官方预处理参数：

```text
Resize size:
256

Center Crop:
224
```

模型实际看到的几何过程：

```text
Original Image
↓
Resize(shorter side = 256)
↓
Center Crop(224 × 224)
↓
Tensor
↓
ImageNet Normalize
```

因此 Ground Truth 也必须使用相同的：

```text
Resize
+
Center Crop
```

否则即使最终 Shape 同样是：

```text
[224,224]
```

空间位置也可能不一致。

核心原则：

> Shape 一样，不代表 Geometry 一样。

---

## 5. Prediction

EXP011 已经得到 `broken_small/000.png` 的 Top-5 Patch：

```text
(17,7)
(18,7)
(19,8)
(19,7)
(19,6)
```

因此预测：

```text
高异常响应主要集中于：

row ≈ 17~19
col ≈ 6~8
```

将 `[28,28]` Anomaly Map 上采样至 `[224,224]` 后：

```text
预计会在图像左下侧局部区域形成明显高响应。
```

如果该高响应区域与 Ground Truth 缺陷区域明显重合，则支持：

> Patch-level representation 和 nearest-neighbor anomaly score 确实捕捉到了局部小缺陷。

如果高响应区域与 Ground Truth 明显不重合，则说明：

```text
高分可能来自：
正常纹理
反光
边缘
结构变化
或其他正常视觉变化
```

此时不能宣称定位成功。

---

## 6. Core Code

### 6.1 Prepare Model Input for Visualization

模型输入：

```python
x = preprocess(image)
```

已经包含：

```text
Resize
Center Crop
Tensor Conversion
ImageNet Normalization
```

为了显示模型真正看到的图像，需要反归一化：

```python
display = (
    x * std + mean
).clamp(0, 1)
```

原始 PyTorch Layout：

```text
[C,H,W]
```

转为 Matplotlib Layout：

```python
display = display.permute(
    1,
    2,
    0,
)
```

最终：

```text
[224,224,3]
```

---

### 6.2 Prepare Ground Truth

Ground Truth 读取：

```python
mask = Image.open(
    mask_path
).convert("L")
```

转换 Tensor：

```python
mask = TF.pil_to_tensor(
    mask
).float() / 255.0
```

然后使用与模型输入相同的几何变换：

```python
mask = TF.resize(
    mask,
    size=preprocess.resize_size,
    interpolation=InterpolationMode.NEAREST,
    antialias=False,
)
```

再：

```python
mask = TF.center_crop(
    mask,
    output_size=preprocess.crop_size,
)
```

最后二值化：

```python
mask = (
    mask > 0.5
).float()
```

---

### 6.3 Why Ground Truth Uses Nearest Neighbor

Ground Truth Mask 是离散标签：

```text
0 = normal
1 = defect
```

如果使用 Bilinear Resize，可能产生：

```text
0.13
0.47
0.72
```

这类中间值。

因此 Mask Resize 使用：

```text
Nearest Neighbor
```

而自然 RGB 图像通常可以使用：

```text
Bilinear
Bicubic
```

---

### 6.4 Upsample Anomaly Map

低分辨率 Patch Map：

```text
[28,28]
```

先扩维：

```python
x = anomaly_map.unsqueeze(0).unsqueeze(0)
```

得到：

```text
[1,1,28,28]
```

再：

```python
upsampled = F.interpolate(
    x,
    size=(224,224),
    mode="bilinear",
    align_corners=False,
)
```

最终：

```text
[224,224]
```

注意：

> Upsampling 只是在空间上插值放大，不会产生新的定位信息。

真实定位分辨率仍然受原始：

```text
28×28
```

Feature Map 限制。

---

### 6.5 Visualization Normalization

为了画热力图：

```python
normalized = (
    score_map - minimum
) / (
    maximum - minimum + 1e-8
)
```

得到：

```text
[0,1]
```

这个 Normalize：

```text
只用于 Visualization
```

不能用于不同图片之间的定量 Score 比较。

正确原则：

```text
Evaluation:
Raw Anomaly Score

Visualization:
Min-Max Normalized Score
```

---

### 6.6 Ground Truth Bounding Box

首先：

```python
positions = torch.nonzero(
    gt_mask > 0.5,
    as_tuple=False,
)
```

Tensor 图像坐标顺序：

```text
[y,x]
```

再求：

```text
x_min
y_min
x_max
y_max
```

最终得到 GT Bounding Box。

---

### 6.7 Patch Position → Input Coordinate

Feature Map：

```text
28 × 28
```

Model Input：

```text
224 × 224
```

因此：

```text
scale_x = 224 / 28 = 8
scale_y = 224 / 28 = 8
```

Patch Center 近似：

```text
center_x = (col + 0.5) × 8
center_y = (row + 0.5) × 8
```

例如：

```text
(row=17,col=7)
```

得到：

```text
x = 60
y = 140
```

---

## 7. Result

### Model Preprocessing

```text
Resize size:
[256]

Crop size:
[224]
```

---

### Memory Bank

```text
torch.Size([163856, 128])
```

---

### Query Patches

```text
torch.Size([784, 128])
```

---

### Low-resolution Anomaly Map

```text
torch.Size([28, 28])
```

---

### Model Input Visualization

```text
torch.Size([224, 224, 3])
```

---

### Upsampled Anomaly Map

```text
torch.Size([224, 224])
```

---

### Ground Truth

```text
Ground-truth mask shape:
torch.Size([224, 224])

Ground-truth positive pixels:
582
```

整张模型输入包含：

$$
224\times224=50176
$$

pixels。

因此缺陷区域约占：

$$
\frac{582}{50176}
\approx 1.16\%
$$

说明该缺陷确实属于较小的局部异常。

---

### Ground Truth Bounding Box

```text
(x_min, y_min, x_max, y_max)

(41, 134, 67, 165)
```

因此 Ground Truth 大致位于：

```text
x = 41~67
y = 134~165
```

---

## 8. Highest-Scoring Patch

最高异常 Patch：

```text
Patch index:
483

Feature-map position:
(17,7)

Score:
3.906899

Approx input-space center:
(x=60, y=140)
```

Ground Truth：

```text
x = 41~67
y = 134~165
```

因此：

```text
x=60 ∈ [41,67]
y=140 ∈ [134,165]
```

实际程序输出：

```text
Patch center inside GT mask:
True
```

---

## 9. Top-5 Patch / Ground Truth Comparison

| Rank | Patch | Feature Position | Input Center | Score | Center in GT |
|---:|---:|---|---|---:|---|
| 1 | 483 | (17,7) | (60,140) | 3.906899 | True |
| 2 | 511 | (18,7) | (60,148) | 3.702626 | True |
| 3 | 540 | (19,8) | (68,156) | 3.525064 | False |
| 4 | 539 | (19,7) | (60,156) | 3.482851 | True |
| 5 | 538 | (19,6) | (52,156) | 3.354582 | True |

Top-5 中：

```text
4 / 5
```

Patch Center 直接位于 Ground Truth 内部。

Rank 3：

```text
center x = 68
```

而 Ground Truth bbox：

```text
x_max = 67
```

仅相差：

```text
1 pixel
```

因此 Rank 3 虽然中心点没有进入二值 Mask，但实际上位于 Ground Truth 边界紧邻位置。

---

## 10. Receptive Field Important Note

Feature Map：

```text
28×28
```

与 Input：

```text
224×224
```

之间的：

```text
8 pixel
```

表示相邻 Feature Center 的空间步距。

它不代表一个 Layer2 Feature 只观察：

```text
8×8 pixels
```

ResNet18 `layer2` 的一个 Feature 具有明显更大的 receptive field。

因此：

```text
Patch Center outside GT
```

不能直接解释为：

```text
该 Feature 与缺陷无关
```

只要 Feature 的 receptive field 覆盖缺陷区域，该 Patch Feature 仍可能受到异常影响。

---

## 11. Visualization Observation

生成图像包括：

```text
Model Input
Anomaly Map
Image + Anomaly
Ground Truth
```

观察结果：

```text
Anomaly Map 的最强红色区域
```

位于：

```text
瓶口左下区域
```

Ground Truth 缺陷也位于：

```text
瓶口左下区域
```

两者空间上具有明显对应关系。

同时，瓶口圆环其他区域仍然出现部分：

```text
cyan
green
yellow
```

响应。

说明当前 Layer2 + NN 方法不仅对缺陷响应，也会对一些正常视觉变化产生较高分数。

潜在原因包括：

```text
反光
边缘
正常纹理变化
局部结构变化
对齐变化
```

这些可以作为后续 Failure Analysis 的候选。

---

## 12. Prediction vs Result

### Prediction 1

预测：

```text
broken_small 高分区域
应集中在局部空间范围内
```

实际：

```text
Top-5:

(17,7)
(18,7)
(19,8)
(19,7)
(19,6)
```

结论：

```text
PASS
```

---

### Prediction 2

预测：

```text
高响应区域应与真实缺陷存在明显空间对应。
```

实际：

```text
Top-1 center inside GT:
True

Top-2 center inside GT:
True

Top-3 center:
距 GT bbox 边界约 1 pixel

Top-4 center inside GT:
True

Top-5 center inside GT:
True
```

结论：

```text
PASS
```

---

## 13. Analysis

本实验建立了以下完整因果链：

```text
broken_small defect
↓
局部视觉结构发生变化
↓
ResNet18 Layer2 Patch Feature 改变
↓
无法在 Normal Memory Bank 中找到足够相似的正常 Patch
↓
Nearest Neighbor Distance 增大
↓
Patch Anomaly Score 增大
↓
高分 Patch 出现空间聚集
↓
该高分区域与 Ground Truth 缺陷区域明显对应
```

这为 Patch-level anomaly detection 捕捉局部小缺陷提供了直接空间证据。

---

## 14. Limitation

当前实验只验证了：

```text
broken_small/000.png
```

一个样本。

因此目前可以说：

> 在该样本上，Patch-level NN anomaly scoring 产生了与真实缺陷明显对应的局部高响应区域。

但不能直接说：

```text
整个 bottle 数据集定位性能很好
```

也不能说：

```text
像素级分割性能很好
```

因为当前尚未进行：

```text
Pixel AUROC
PRO
IoU
Dice
Pixel Threshold
Full Test Set Evaluation
```

此外：

```text
28×28 Feature Map
+
CNN Receptive Field
+
Bilinear Upsampling
```

决定了 Anomaly Map 边界会比真实缺陷更模糊。

---

## 15. Saved Artifacts

Visualization：

```text
results/exp012/
broken_small_000_anomaly_map.png
```

Raw Scores：

```text
results/exp012/
broken_small_000_scores.pt
```

保存内容：

```text
patch_scores
anomaly_map
upsampled_map
gt_mask
nearest_indices
```

---

## 16. Conclusion

EXP012 成功完成：

```text
Patch Score Map
↓
Spatial Reconstruction
↓
224×224 Upsampling
↓
Ground Truth Alignment
↓
Visualization
↓
Spatial Verification
```

结果证明：

```text
Top anomaly response
与
broken_small Ground Truth
具有明显空间对应
```

特别是：

```text
Top-5 Patch Centers:
4/5 directly inside GT

Remaining 1:
adjacent to GT boundary
```

因此当前证据支持：

> Patch-level Feature 能够保留局部小缺陷信息，并在该样本上提供有效的定性缺陷定位。

---

## 17. New Questions

1. Memory Bank 是否必须保留全部 163856 个 Patch？
2. 删除部分正常 Patch 后，Anomaly Map 是否还能保持稳定？
3. Memory Bank Size 如何影响正常模式覆盖？
4. Memory Bank Reduction 是否会提高 False Positive Risk？
5. 如何在性能和计算量之间取得平衡？
6. Random Sampling 和 Coreset Sampling 有什么差异？
7. 如何扩展到整个 Test Set？

---

## 18. Next Step

下一实验：

```text
EXP013 - Memory Bank Size Ablation
```

目标：

```text
Full Memory
vs
10% Random Memory
vs
1% Random Memory
```

比较：

```text
Score Distribution
Score-map Correlation
Top-K Spatial Stability
Pairwise Workload
Memory Coverage
```

---

## Status

```text
Geometry Alignment:
PASS

Anomaly Map Visualization:
PASS

Ground Truth Comparison:
PASS

Top-K Spatial Verification:
PASS

Qualitative Localization:
PASS

EXP012:
PASS

Current Learning Unit:
CONTINUE
```