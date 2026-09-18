## Result

Input:
[1, 3, 224, 224]

Layer2:
[1, 128, 28, 28]

Layer3:
[1, 256, 14, 14]

Layer2 patches:
[1, 784, 128]

Layer3 patches:
[1, 196, 256]

Spatial position:
(row=10, col=5)

Expected patch index:
10 * 28 + 5 = 285

Maximum absolute error:
0.0

## Prediction vs Result

预测与实验结果一致。

Feature Map 中位置 (10,5) 的 128 维特征，
与 flatten 后 patch index=285 的特征完全一致。

## Analysis

permute 只改变维度顺序：

[B,C,H,W]
→
[B,H,W,C]

reshape 将二维空间位置展平：

[B,H,W,C]
→
[B,H*W,C]

整个过程没有执行特征平均或数值变换，
因此局部 Feature 与空间位置仍然可以相互对应。

## Conclusion

Patch Feature 保留了局部空间表示。

相比 Global Average Pooling 得到单个全局向量，
Patch representation 可以分别分析不同空间区域，
为后续局部异常评分和 Anomaly Map 提供基础。