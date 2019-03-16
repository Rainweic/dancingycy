# YCY Dance Now

## Pipeline
* 本项目实现了一种基于pix2pixHD生成对抗网络和时域空间平滑的运动转移方法，用户给定一段跳舞的源视频，我们可以该表演动作转移到另一主体——杨超越，从而生成杨超越完成该舞蹈的视频。我们使用姿势检测作为用户舞蹈和杨超越舞蹈之间的中间表示，学习从姿势图像到杨超越主体外观的映射。为了完成这项任务，我们将项目流程划分为三个阶段：姿态检测、全局姿态归一化以及从标准化姿态图到杨超越图像的映射。在姿态检测阶段，我们使用SOTA姿势检测器来对定自源视频每一帧创建姿态图。全局姿态标准化阶段考虑了源体姿态和杨超越姿态的形状与镜头位置之间的差异。最后，我们设计了一个基于pix2pixHD生成对抗网络的端到端系统来学习从标准化的姿态图到用于对抗训练的杨超越的图像的映射。为保证结果视频的真实性，我们结合了视频时域空间的平滑方法和真实感的面部合成方法对视频结果进行优化。
* 为了展示生成效果，我们搭建了用于展示的网站：http://www.dancingycy.top/#/。

## Cite
* [pytorch-EverybodyDanceNow](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow)
* [EverybodyDanceNow_reproduce_pytorch](https://github.com/CUHKSZ-TQL/EverybodyDanceNow_reproduce_pytorch)
