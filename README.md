# CamvidUnetW
camvid Unet learn demo  简单的camvid Unet学习demo

数据集：Camvid http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

在GTX1650 进行训练，受限于环境，差不多把除了最基本的训练都砍了

loaddata.py : Dataset自定义、数据集预处理

unet.py : Unet网络定义，源于 https://github.com/milesial/Pytorch-UNet

train.py : 训练相关函数定义

predict.py ： 预测、图像显示相关


2023.9.9 重大bug：loaddata文件中cv2openimg函数 使用了cv2.reshape，这将导致复原的图像非原图、训练结果出现重大问题
                  应修改为img.transpose(2,0,1) (已修改)
