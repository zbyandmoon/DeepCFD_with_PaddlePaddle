### 【论文复现赛】DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks

* 队伍名称：zbyandmoon

* 论文题目：DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks (https://arxiv.org/abs/2004.08826)

* 模型描述：文中使用的DeepCFD网络基本结构为有3个输入和3个输出的U-net型网络。该模型输入为计算域中几何体的符号距离函数（SDF）、计算域边界的SDF和流动区域的标签；输出为流体的x方向速度、y方向速度以及流体压强。该模型的基本原理就是利用编码器部分的卷积层将3个输入下采样，变为中间量，然后使用相同结构的解码器中的反卷积层将中间量上采样为3个流体物理量输出。

* 预测结果：

  ① 原文结果：

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220729145921864.png" alt="image-20220729145921864" style="zoom:50%;" />

  ② 跑通原文参考代码的结果：

  **训练截图：**（注：原文一共也是训练了1000个Epochs，这里限于篇幅仅仅截了最后一步输出，可以看到输出的MSE (Ux=0.7645，Uy=0.1972，p=1.0410，Total=2.0026）在原文范围之内）

  ![image](https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/blob/main/pytorch/Run/%E8%AE%AD%E7%BB%83%E6%88%AA%E5%9B%BE.png)

  **程序最后输出的CFD和CNN的对比图：**

  ![image-20220729145007729](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220729145007729.png)

  

* 复现地址：https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/tree/main/pytorch