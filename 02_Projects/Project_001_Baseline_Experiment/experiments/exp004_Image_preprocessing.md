# EXP004 Image Preprocessing


## 实验目标

验证图像从原始像素到模型输入Tensor的预处理流程。


## 输入

RGB图片:

400×400×3


## 处理流程

Image

↓

Resize(224×224)

↓

ToTensor

↓

Normalize


## 我的预测

1. 输出尺寸：

3×224×224


2. 数据类型：

float32


3. Normalize后：

数据范围出现负数。


## 实际结果

输出：

Tensor shape:

torch.Size([3,224,224])


dtype:

torch.float32


min:

-2.10


max:

2.64


## 分析

Resize完成尺寸统一。

ToTensor完成：

uint8 → float32

HWC → CHW


Normalize使数据分布中心化，

因此输出范围不再是0-1。


## 结论

完成视觉模型输入预处理流程。

理解了Resize、ToTensor和Normalize的作用。