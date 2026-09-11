# EXP005 Feature Extractor


## 实验目标

验证预训练ResNet18是否能够将工业图片转换为特征向量。


## 实验假设

输入图片Tensor:

[8,3,224,224]

经过ResNet18 backbone后，

输出:

[8,512]


## 实验结果

Input:

torch.Size([8,3,224,224])


Feature:

torch.Size([8,512])


## 分析

ResNet18成功完成图像特征提取。

移除分类头后，可以作为异常检测中的feature extractor。


## 结论

完成Image到Feature的转换。

# EXP005 Normal Feature Center


## 实验目标

建立工业异常检测Baseline中的正常特征分布中心。


## 实验数据

Dataset:

MVTec AD

Category:

bottle


Training:

bottle/train/good

Images:

209


## 实验方法

使用预训练ResNet18作为Feature Extractor。


输入：

[batch,3,224,224]


输出：

[batch,512]


对全部正常样本feature求均值：

Center = mean(features)


## 实验结果

Normal Feature Center:

torch.Size([512])


## 分析

成功将209张正常工业图像转换为512维特征空间，并建立正常样本中心。


## 结论

完成Baseline异常检测中的Normal Modeling阶段。

下一步进行Anomaly Score计算。