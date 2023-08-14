"""predict s&showpics"""
import train as tr
from torchvision import transforms
import loaddata as ld
from torch.utils.data import Dataset,DataLoader
import unet
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

def GetColorsArray(dimenshions=32):
    """Load the mapping that associates pascal classes with label colors"""
    ColorsArray = np.empty((dimenshions,3))
    for i in range(dimenshions):
        ColorsArray[i]=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
    return ColorsArray

'''
    np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])               # 根据自己的类别，自定义颜色
'''

def GetLabelsArray(path):
    with open(path, 'r') as fh:
        LabelsArray = []
        for line in fh:
            line1 = line.rstrip()  # 删除末尾空
            LabelsArray.append(line1)
    return LabelsArray

def ShowPredicts(net,n,val_loader,Labels,Colors):
    """use plt auto to GRB?"""
    for imgs, masks in val_loader:
        break
    imgs = imgs.float()
    preds = net(imgs)
    masks = masks.permute(0, 2, 3, 1)
    imgs=imgs.permute(0,2,3,1)
    preds = torch.argmax(preds, dim=1, keepdim=True) #predstensor 2 pics
    preds = preds.permute(0, 2, 3, 1)
    #tensor2numpy
    imgs=imgs.numpy()
    preds=preds.numpy()
    masks=masks.numpy()
    plt.figure(1)
    for i in range(n):
        plt.subplot(n, 3, (i + 1) * 3 - 2)
        ###原图显示有bug
        plt.imshow(imgs[i]/255)
        plt.xticks([])
        plt.yticks([])
        plt.title('Img')
        plt.subplot(n, 3, (i+1)*3-1)
        plt.imshow(masks[i])
        plt.xticks([])
        plt.yticks([])
        plt.title('True')
        plt.subplot(n, 3, (i + 1)*3)
        plt.imshow(preds[i])
        plt.xticks([])
        plt.yticks([])
        plt.title('Pred')
    plt.show()

if __name__ == '__main__':
    datapath = r'E:\Desktop\unet\camvid'
    initmodel = r'E:\Desktop\unet\model\Epoch107Lr0.0001Bz8loss0.11716232792962165.pth'
    batchsize = 10
    n=3
    test_data = ld.RoadDataset(txt=datapath + '\\' + 'test_set.txt', transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)
    val_data = ld.RoadDataset(txt=datapath + '\\' + 'val_set.txt', transform=transforms.ToTensor())
    val_loader = DataLoader(dataset=val_data, batch_size=batchsize, shuffle=True)

    net = unet.UNet(n_channels=3, n_classes=32)
    m_state_dict = torch.load(initmodel)
    net.load_state_dict(m_state_dict)
    net = net.to('cpu')
    #ac = tr.EvalAccuracy(net, test_loader, train=False, device=tr.try_gpu())
    #print(ac)
    ColorsArray=GetColorsArray()
    LabelsArray=GetLabelsArray(datapath+'\\codes.txt')


    ShowPredicts(net,n, val_loader, LabelsArray, ColorsArray)





