"""total"""
import loaddata
import unet
import train
import predict
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import time
import numpy as np

datapath = r'E:\Desktop\unet\camvid'    ##dataset path
batchsize = 8
initmodel = r'E:\Desktop\unet\model\Epoch107Lr0.0001Bz8loss0.11716232792962165.pth'
beginnum = 100      ##retrain epochs
lr = 0.0001
numepochs = 20
picchannels=3
predclasses=32
finmodel=r'E:\Desktop\unet\model\Epoch107Lr0.0001Bz8loss0.11716232792962165.pth'
n=3     #predict num

if __name__ == '__main__':
    """1 load data"""
    train_data = loaddata.RoadDataset(txt=datapath + '\\' + 'train_set.txt', transform=transforms.ToTensor())
    test_data = loaddata.RoadDataset(txt=datapath + '\\' + 'test_set.txt', transform=transforms.ToTensor())
    val_data = loaddata.RoadDataset(txt=datapath + '\\' + 'val_set.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batchsize, shuffle=True)
    """2 load net"""
    net = unet.UNet(n_channels=picchannels, n_classes=predclasses)
    if beginnum != 0:
        m_state_dict = torch.load(initmodel)
        net.load_state_dict(m_state_dict)
    net = net.to(train.try_gpu())
    """3 train"""
    timer0 = time.perf_counter()
    trainloss ,testloss,testacc= train.traintqdm(net, train_loader, test_loader, numepochs, be, lr, batchsize, integrity=False,
                          device=train.try_gpu())
    timer1 = time.perf_counter()
    print(f'TotalTime:{timer1 - timer0} s')
    np.set_printoptions(precision=3)
    print('trainloss:', trainloss)
    print('trainacc:', testloss)
    print('testacc:', testacc)
    """4 predict"""
    if finmodel:
        m_state_dict = torch.load(initmodel)
        net.load_state_dict(m_state_dict)
    net = net.to('cpu')
    # ac = train.EvalAccuracy(net, test_loader, train=False, device=tr.try_gpu())
    # print(ac)
    ColorsArray = predict.GetColorsArray()
    LabelsArray = predict.GetLabelsArray(datapath + '\\codes.txt')
    predict.ShowPredicts(net, n, val_loader, LabelsArray, ColorsArray)