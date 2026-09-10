# Image Basics


# 1. 图像在计算机中的表示


人眼看到的是图片。

计算机看到的是数字。


一张图片本质上是一个多维数组。


例如：

灰度图片：

Height × Width


RGB图片：

Height × Width × Channel



计算机通过像素值表示图像信息。


---

# 2. Pixel（像素）


Pixel：

Picture Element。


像素是图像中的最小单位。



每一个Pixel保存图像信息。


例如：

灰度图：

一个像素：

120


表示亮度。


RGB图：

一个像素：

(255,0,0)


表示红色。



---

# 3. RGB Channel


彩色图片通常由三个Channel组成：


R:

Red


G:

Green


B:

Blue



每个Channel表示一种颜色信息。



所以RGB图片：

Channel=3



例如：

一个RGB像素：

(R,G,B)


=(100,150,200)



---

# 4. Image Shape


深度学习中通常表示：


Height × Width × Channel



例如：


640 × 480 × 3



表示：

640:

图片高度 Height



480:

图片宽度 Width



3:

RGB三个颜色通道 Channel



---

# 5. JPG到Tensor流程


图片文件：

.jpg


↓

图像读取


↓

像素矩阵


↓

numpy array


↓

Tensor


↓

输入神经网络



神经网络不能直接理解jpg文件，

只能处理数字张量。



---

# 6. Tensor是什么


Tensor：

张量。


本质：

多维数组。



例如：


一维：

[1,2,3]


二维：

矩阵


三维：

Image:

Height × Width × Channel



深度学习模型通常接收Tensor作为输入。



---

# 7. Dataset作用


Dataset：

数据集合。



视觉任务中：

Dataset通常包含：


Image

+

Annotation



代码层面Dataset负责：


1. 找到图片文件


2. 读取图片


3. 图片预处理


4. 读取标签


5. 返回模型需要的数据



数据流程：


Image

↓

Dataset

↓

DataLoader

↓

Tensor

↓

Model



---

# 8. 为什么Dataset重要


模型性能不仅取决于网络结构，

也取决于数据。


Dataset决定：

- 数据质量
- 数据分布
- 标注质量
- 输入形式



如果Dataset设计错误：

即使模型很强，

结果也可能失败。



---

# 9. 我的理解


图片不是直接输入模型的。

它需要经过：

图片文件

↓

像素矩阵

↓

Tensor

↓

模型


RGB图片由三个Channel组成。

Dataset负责把大量图片和标签组织起来，

提供给模型训练和测试。



视觉AI工程的核心流程：

数据

↓

模型

↓

预测

↓

评价

↓

优化


