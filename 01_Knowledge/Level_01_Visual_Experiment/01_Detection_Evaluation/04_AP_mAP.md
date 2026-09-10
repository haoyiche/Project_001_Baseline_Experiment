# AP 与 mAP


# 1. 为什么需要 AP

目标检测模型评价时：

Precision：

表示预测结果的可靠程度。

Recall：

表示真实目标被发现的比例。



但是：

Precision 和 Recall 关注不同方向。


Precision高：

模型比较保守，
误检少。


Recall高：

模型比较激进，
漏检少。


因此不能只看单个指标。

需要综合评价：

AP。



---

# 2. Confidence Threshold


目标检测模型输出：

- 类别
- Bounding Box
- Confidence


Confidence表示：

模型认为该预测可信的程度。


例如：

预测结果：

Box A:

confidence=0.95


Box B:

confidence=0.80


Box C:

confidence=0.40



设置阈值：

confidence threshold=0.5


保留：

A、B


删除：

C



---

# 3. Confidence Threshold 对 Precision 和 Recall 的影响


降低confidence threshold：

例如：

0.8 → 0.3


更多预测框被保留。


结果：

找到更多目标。


通常：

Recall ↑


但是：

误检增加。


通常：

Precision ↓



---

提高confidence threshold：

例如：

0.3 → 0.8


预测更加严格。


结果：

误检减少。


通常：

Precision ↑


但是：

可能漏掉更多目标。


通常：

Recall ↓



---

# 4. Precision-Recall Curve


当confidence threshold不断变化时：

Precision和Recall也会变化。


将不同threshold下：

Precision

Recall


绘制成曲线：

称为：

PR Curve


横轴：

Recall


纵轴：

Precision



模型越好：

曲线越靠近右上方。



---

# 5. AP（Average Precision）


AP：

Average Precision


中文：

平均精度。



定义：

AP表示：

Precision-Recall曲线下面积。



公式：


AP = Area(PR Curve)



含义：

如果模型在不同Recall情况下，

仍然保持较高Precision，

那么：

AP较高。



---

# 6. mAP（mean Average Precision）


目标检测通常包含多个类别。


例如：

工业检测：

类别：

- 裂纹
- 划痕
- 污渍
- 缺陷


每个类别都有自己的AP。



例如：

|类别|AP|
|-|-|
|裂纹|0.92|
|划痕|0.85|
|污渍|0.80|



mAP：

表示所有类别AP的平均。



公式：

mAP =
(AP1 + AP2 + ... + APn) / n



---

# 7. mAP50


mAP50表示：

IoU阈值为0.5时计算mAP。



要求：

预测框与真实框：

IoU >= 0.5



特点：

要求相对宽松。



---

# 8. mAP50-95


COCO评价标准。


计算多个IoU阈值：

0.50

0.55

0.60

...

0.95



然后求平均。



相比mAP50：

要求更严格。



---

# 9. mAP50 和 mAP50-95区别


mAP50：

关注：

是否大致找到目标。



mAP50-95：

更加关注：

定位精度。



通常：

mAP50 > mAP50-95



---

# 10. 工业视觉中的理解


mAP不是越高越代表一定适合部署。


需要结合任务。


例如：


缺陷筛选：

关注：

Recall

避免漏检。


机器人抓取：

关注：

IoU

保证定位准确。


高速生产线：

还需要考虑：

- 推理速度
- 显存
- 延迟



---

# 11. 完整检测评价流程


输入图片

↓

模型预测多个框

↓

Confidence筛选

↓

IoU匹配Ground Truth

↓

得到TP / FP / FN

↓

计算Precision / Recall

↓

绘制PR Curve

↓

计算AP

↓

多个类别平均

↓

得到mAP



---

# 12. 我的理解


mAP不是模型能力本身，

而是模型在检测任务上的综合评价。


Precision关注：

预测出来是否可靠。


Recall关注：

真实目标是否漏掉。


IoU关注：

位置是否准确。


AP综合Precision和Recall变化。


mAP综合多个类别表现。


