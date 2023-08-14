"""train&test"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import loaddata as ld
import unet
from torchvision import transforms
from tqdm import tqdm
import time
import os

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def EvalAccuracy(net, data_iter,train=False, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    torch.cuda.empty_cache()
    if train:
        net.train()
    else:
        net.eval()  # 设置为评估模式
    net.to(device)
    acc=0
    num_batches = len(data_iter)
    for batch, (image, segment_image) in enumerate(data_iter):
        image, segment_image = image.to(device), segment_image.to(device)
        image = image.float()
        pred=net(image)
        pred=torch.argmax(pred,dim=1,keepdim=True)
        ans=segment_image.eq(pred)
        ac=torch.sum(ans)
        tc=ans.numel()
        acc+=ac.item()/tc
    acc=acc/num_batches
    return acc

def SaveData(path='traindata.txt',epoch=0,trainloss=0,testloss=-1,testacc=-1):
    """ loss/acc save txt """
    '''
    if os.path.exists(path + '\\' + 'test_set.txt'):
        os.remove(path + '\\' + 'test_set.txt')
    '''
    with open(path, mode='a') as file:
        file.write('Epochs:'+str(epoch)+','+'Trainloss:'+str(trainloss)+',')
        if testloss!=-1:
            file.write('Testloss:'+str(testloss)+ ',')
        if testacc != -1:
            file.write('Testacc:' + str(testacc) + ',')
        file.write('\n')
                # 为了以后好用，修改了这里，将' '改成了','，另外路径加了sub_dir

def train(net,train_iter,num_epochs,beginepochs,lr,bs,device= try_gpu()):
    if beginepochs == 0:
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)  # weight_decay=weight_decay,,  foreach=True
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)#####
    loss = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    num_batches = len(train_iter)
    trainloss = np.empty(num_epochs, dtype=float)
    trainacc = np.empty(num_epochs, dtype=float)
    testacc = np.empty(num_epochs, dtype=float)
    for epoch in range(num_epochs):
        print('training epoch:',epoch+1)
        epoch_loss = 0
        timer0 = time.perf_counter()
        torch.cuda.empty_cache()
        net.train()
        for batch, (image, segment_image) in enumerate(train_iter):
            optimizer.zero_grad()
            image, segment_image = image.to(device), segment_image.to(device)
            image = image.float()  #####
            torch.cuda.empty_cache()
            out_image = net(image)
            segment_image = torch.argmax(segment_image, dim=1)  ##?
            train_loss = loss(out_image, segment_image.long())
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        print(epoch_loss)
        timer1 = time.perf_counter()
        print(f'EpochTime:{timer1-timer0} s')
        trainloss[epoch]=epoch_loss
        pth = 'model/' + 'Epoch' + str(epoch + 1 + beginepochs) + 'Lr' + str(lr) + 'Bz' + str(bs) + '.pth'
        torch.save(net.state_dict(), pth)
    return train_loss


def traintqdm(net,train_iter,test_iter,num_epochs,beginepochs,lr,bs,integrity=False,device=try_gpu()):
    if beginepochs == 0:
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # weight_decay=weight_decay,,  foreach=True
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)#####
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    trainloss = np.empty(num_epochs, dtype=float)
    testacc = np.empty(num_epochs, dtype=float)
    testloss = np.empty(num_epochs, dtype=float)
    for epoch in range(num_epochs):
        epoch_loss=0
        timer0=time.perf_counter()
        #torch.cuda.empty_cache()
        net.train()
        with tqdm(iterable=train_iter, unit="batch") as tepoch:  # 1. 定义进度条
            for batch, (image, segment_image) in enumerate(train_iter): # 2. 设置迭代器
                tepoch.set_description(f"Epoch {epoch+1}")  # 3. 设置开头
                #optimizer.zero_grad()
                image, segment_image = image.to(device), segment_image.to(device)
                image = image.float()  #####
                #torch.cuda.empty_cache()
                out_image = net(image)
                segment_image=segment_image.squeeze()#降维##减小维度
                train_loss = loss(out_image, segment_image.long())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item()
                tepoch.set_postfix(trainloss=train_loss.item())  # 4. 设置结尾 , accuracy='{:.3f}'.format(accuracy)
                tepoch.update() # 5.更新进度条
            if integrity:
                tepoch_loss=0
                for batch, (image, segment_image) in enumerate(test_iter):  # 2. 设置迭代器
                    tepoch.set_description(f"Epoch {epoch + 1}")  # 3. 设置开头
                    # optimizer.zero_grad()
                    image, segment_image = image.to(device), segment_image.to(device)
                    image = image.float()  #####
                    # torch.cuda.empty_cache()
                    out_image = net(image)
                    segment_image = segment_image.squeeze()  # 降维##减小维度
                    test_loss = loss(out_image, segment_image.long())
                    optimizer.zero_grad()
                    test_loss.backward()
                    optimizer.step()
                    tepoch_loss += test_loss.item()
                    tepoch.set_postfix(tsetloss=test_loss.item())  # 4. 设置结尾 , accuracy='{:.3f}'.format(accuracy)
                    tepoch.update()  # 5.更新进度条
                testloss[epoch]=tepoch_loss/num_batches
                testacc[epoch] = EvalAccuracy(net, test_iter, train=False, device=device)
        print(epoch_loss/num_batches)
        timer1 =time.perf_counter()
        print(f'EpochTime:{timer1-timer0} s')
        trainloss[epoch]=epoch_loss/num_batches

        pth = ('model/' +
               'Epoch' + str(epoch + 1 + beginepochs) + 'Lr' + str(lr) + 'Bz' + str(bs) +
               'Ls'+str(round(trainloss[epoch], 3))+ '.pth')
        torch.save(net.state_dict(), pth)
    return trainloss,testloss,testacc







if __name__ == '__main__':
    datapath=r'E:\Desktop\unet\camvid'
    batchsize=8
    initmodel = r'E:\Desktop\unet\model\Epoch107Lr0.0001Bz8loss0.11716232792962165.pth'
    be=100
    lr=0.0001
    numepochs=20

    net = unet.UNet(n_channels=3, n_classes=32)
    if be!=0:
        m_state_dict = torch.load(initmodel)
        net.load_state_dict(m_state_dict)
    net = net.to(try_gpu())
    train_data=ld.RoadDataset(txt=datapath + '\\' + 'train_set.txt',transform=transforms.ToTensor())
    test_data=ld.RoadDataset(txt=datapath + '\\' + 'test_set.txt',transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)
    #print(len(train_loader))
    timer0 = time.perf_counter()
    trainloss=traintqdm(net,train_loader,test_loader, numepochs, be, lr,batchsize,integrity=False,device=try_gpu())
    timer1=time.perf_counter()
    #plt.show()
    print(f'TotalTime:{timer1-timer0} s')
    print(trainloss)
    '''
    np.set_printoptions(precision=3)
    print('trainloss:', trainloss)
    print('trainacc:', trainacc)
    print('testacc:', testacc)
    '''

