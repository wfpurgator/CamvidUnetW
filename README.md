# CamvidUnetW
camvid Unet learn demo  简单的camvid Unet学习demo

数据集：Camvid http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

在GTX1650 进行训练，受限于环境，差不多把除了最基本的训练都砍了

loaddata.py : Dataset自定义、数据集预处理

unet.py : Unet网络定义，源于 https://github.com/milesial/Pytorch-UNet

train.py : 训练相关函数定义

predict.py ： 预测、图像显示相关
