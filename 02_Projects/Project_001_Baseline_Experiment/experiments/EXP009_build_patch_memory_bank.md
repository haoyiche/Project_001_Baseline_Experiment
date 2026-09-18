# EXP009 - Build Normal Patch Memory Bank

## 1. Goal

本实验的目标是：

将 MVTec AD `bottle/train/good` 中所有正常训练图像转换为 ResNet18 `layer2` 的局部 Patch Feature，并将全部正常 Patch 组合成一个 **Normal Patch Memory Bank**。

本实验主要验证以下问题：

- 一张正常图像能够产生多少个 Patch Feature。
- 209 张正常训练图片最终会产生多少个正常 Patch。
- Memory Bank 的 Shape 是否符合理论预测。
- Memory Bank 与 EXP008 中的单图 Patch Feature 是否保持一致。

本实验暂时不进行异常检测，也不计算测试样本的异常分数。

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

Device:
CPU

Dataset:
MVTec AD

Category:
bottle

Split:
train/good
```

---

## 3. Input

正常训练图片数量：

```text
209
```

输入图片经过 ResNet18 官方预处理后：

```text
[3, 224, 224]
```

DataLoader 参数：

```text
batch_size = 16
shuffle = False
num_workers = 0
```

---

## 4. Parameters

使用的 Feature Layer：

```text
ResNet18 layer2
```

Layer2 Feature Map：

```text
[B, 128, 28, 28]
```

每张图片的空间位置数量：

$$
28 \times 28 = 784
$$

因此：

```text
Patches per image = 784
Patch dimension = 128
Normal images = 209
```

---

## 5. Prediction

每张正常图片产生：

$$
28 \times 28 = 784
$$

个 Patch。

因此全部正常训练图片产生：

$$
209 \times 784 = 163856
$$

个 Patch。

每个 Patch Feature 为 128 维，因此预测：

```text
Memory Bank Shape:

[163856, 128]
```

### 最后一个 Batch

因为：

$$
209 = 16 \times 13 + 1
$$

所以预测：

```text
Batch 1 ~ 13:
16 images

Batch 14:
1 image
```

最后一个 Batch 应产生：

```text
[784, 128]
```

### Memory Usage

Memory Bank 中共有：

$$
163856 \times 128
$$

个 `float32`。

每个 `float32` 占 4 bytes，因此预计：

$$
163856 \times 128 \times 4
$$

约为：

```text
80 MiB
```

---

## 6. Core Code

### 6.1 Extract Layer2 Feature

```python
def extract_layer2_feature(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)

    return x
```

输入：

```text
[B, 3, 224, 224]
```

输出：

```text
[B, 128, 28, 28]
```

---

### 6.2 Feature Map → Patch Features

```python
patches = feature_map.permute(0, 2, 3, 1)

patches = patches.reshape(
    batch_size,
    height * width,
    channels,
)
```

Shape 变化：

```text
[B, 128, 28, 28]

↓

[B, 28, 28, 128]

↓

[B, 784, 128]
```

这里没有计算新的 Feature。

`permute()` 只是调整维度顺序，`reshape()` 将二维空间位置展开为 Patch Index。

---

### 6.3 Merge Batch and Patch Dimensions

```python
patches = patches.reshape(
    batch_size_actual * num_patches,
    feature_dim,
)
```

例如 Batch Size 为 16：

```text
[16, 784, 128]

↓

[12544, 128]
```

也就是：

$$
16 \times 784 = 12544
$$

个正常 Patch Feature。

---

### 6.4 Store Patch Features

```python
patches = patches.cpu()

memory_bank_parts.append(patches)
```

如果模型运行在 GPU：

```text
GPU:
负责 Feature Extraction

CPU:
负责保存长期 Memory Bank
```

这样可以避免随着 Batch 增加不断占用 GPU 显存。

本次实验实际运行在 CPU，因此 `.cpu()` 不会发生真正的数据搬迁，但代码逻辑仍然适用于之后的 CUDA 环境。

---

### 6.5 Build Final Memory Bank

```python
memory_bank = torch.cat(
    memory_bank_parts,
    dim=0,
)
```

每个 Batch 产生：

```text
[12544, 128]
```

最后一个 Batch：

```text
[784, 128]
```

沿第 0 维进行拼接，最终得到：

```text
[163856, 128]
```

---

## 7. Result

实际运行环境：

```text
Device: cpu

Normal training images: 209
Batch size: 16
Number of batches: 14
```

前 13 个 Batch：

```text
feature map:
(16, 128, 28, 28)

patches:
(12544, 128)
```

最后一个 Batch：

```text
Batch 14/14

images processed:
209/209

feature map:
(1, 128, 28, 28)

patches:
(784, 128)
```

最终 Memory Bank：

```text
Memory bank shape:
torch.Size([163856, 128])

Memory bank dtype:
torch.float32

Memory bank device:
cpu

Memory bank size:
80.01 MiB
```

Verification：

```text
Expected shape:
(163856, 128)

Actual shape:
(163856, 128)

Shape correct:
True
```

第一条 Memory Bank Feature：

```text
tensor([
    0.2434,
    0.0000,
    0.4600,
    0.3197,
    0.3047,
    1.9978,
    0.0000,
    0.7659,
    1.1229,
    0.3553
])
```

---

## 8. Prediction vs Result

### Prediction 1：正常图片数量

预测：

```text
209
```

实际：

```text
209
```

结果：

```text
PASS
```

### Prediction 2：每张图片 Patch 数量

预测：

```text
784
```

实际：

```text
28 × 28 = 784
```

结果：

```text
PASS
```

### Prediction 3：Memory Bank Shape

预测：

```text
[163856, 128]
```

实际：

```text
[163856, 128]
```

结果：

```text
PASS
```

### Prediction 4：最后一个 Batch

预测：

```text
1 image
784 patches
```

实际：

```text
feature map:
(1, 128, 28, 28)

patches:
(784, 128)
```

结果：

```text
PASS
```

### Prediction 5：Memory Usage

预测：

```text
约 80 MiB
```

实际：

```text
80.01 MiB
```

结果：

```text
PASS
```

---

## 9. Analysis

### 9.1 Memory Bank 的本质

最终：

$$
M \in \mathbb{R}^{163856 \times 128}
$$

其中每一行：

$$
m_i \in \mathbb{R}^{128}
$$

代表某一张正常训练图片中的某一个局部 Patch Feature。

因此可以把 Memory Bank 理解为：

> 正常局部视觉模式数据库。

---

### 9.2 与 Global Baseline 的区别

之前的 Global Feature Baseline：

```text
Image
↓
512-d Global Feature
↓
Normal Images 求平均
↓
一个 Normal Center
```

最终只保留：

```text
[512]
```

当前 Patch 方法：

```text
Image
↓
Layer2 Feature Map
↓
784 Patch Features
↓
所有正常图片 Patch 全部保留
↓
Normal Memory Bank
```

最终：

```text
[163856, 128]
```

最大的区别是：

```text
Global Center:
压缩正常模式

Patch Memory Bank:
保留不同正常局部模式
```

---

### 9.3 为什么不直接求平均

正常图片中存在不同的局部视觉模式，例如：

```text
瓶口
瓶身
边缘
背景
高光区域
```

如果将它们全部压缩成一个平均 Feature，不同正常模式之间的差异可能被抹平。

Memory Bank 则保留这些不同的正常局部模式。

---

### 9.4 EXP008 与 EXP009 的一致性

EXP008 第一张图片的第一个 Layer2 Patch 前 10 个值为：

```text
0.2434
0.0000
0.4600
0.3197
0.3047
1.9978
0.0000
0.7659
1.1229
0.3553
```

EXP009 Memory Bank 第一个 Patch 的值完全一致。

因此可以验证：

```text
EXP008 单图 Patch Extraction

和

EXP009 Dataset Memory Bank
```

使用的是一致的 Feature Extraction 和 Patch 排列逻辑。

---

## 10. Conclusion

EXP009 成功构建了 MVTec AD `bottle/train/good` 的 Normal Patch Memory Bank。

最终结果：

```text
Normal Images:
209

Patches Per Image:
784

Patch Feature Dimension:
128

Total Normal Patches:
163856

Memory Bank Shape:
[163856, 128]

Memory Bank Size:
80.01 MiB
```

完整数据流：

```text
Normal Images
↓
ResNet18 layer2
↓
Feature Map
↓
Patch Features
↓
Flatten Batch/Patch
↓
Concatenate
↓
Normal Patch Memory Bank
```

本实验为下一步 Nearest Neighbor Patch Anomaly Detection 建立了正常特征库。

---

## 11. New Questions

1. 一个测试 Patch 应该如何与 Memory Bank 比较？
2. 如何定义 Patch Anomaly Score？
3. 最近邻距离能否反映异常程度？
4. Memory Bank 很大时，距离计算成本如何控制？
5. 是否真的需要永久保存全部 163856 个 Patch？

---

## 12. Next Step

下一实验：

```text
EXP010 - Single Patch Nearest Neighbor
```

目标：

```text
Query Patch
↓
Normal Memory Bank
↓
Euclidean Distance
↓
Nearest Neighbor
↓
Minimum Distance
↓
Patch Anomaly Score
```

---

## Status

```text
EXP009:
PASS
```