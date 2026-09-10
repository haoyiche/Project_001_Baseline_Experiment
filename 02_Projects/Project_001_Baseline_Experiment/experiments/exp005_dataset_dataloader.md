# EXP005 Dataset and DataLoader


## 实验目标

建立视觉数据从图片文件到Batch Tensor的完整加载流程。


## 实验内容

实现：

Dataset

DataLoader

Transform


完成：

图片读取

↓

预处理

↓

Tensor转换

↓

Batch生成



## 数据

图片数量：

4


图片尺寸：

400×400 RGB



## 实验预测


Dataset输出:

[3,224,224]


DataLoader输出:

[B,3,224,224]



## 实际结果


Dataset:

Dataset size:

4


Single image:

torch.Size([3,224,224])


dtype:

torch.float32



DataLoader:

torch.Size([2,3,224,224])


torch.Size([2,3,224,224])



## 分析


Dataset负责单个样本读取。


DataLoader负责：

batch组织

shuffle

数据加载。


最终形成模型标准输入：

Batch × Channel × Height × Width



## 结论


完成视觉模型数据输入流水线。
