"""data&dataloader&datasets&test"""
import os
import random
import cv2
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import math
from torchvision import transforms

size0=240

def DataDiv(path,imgpath):
    # 文件存在需先移除
    if os.path.exists(path + '\\' + 'test_set.txt'):
        os.remove(path + '\\' + 'test_set.txt')
    if os.path.exists(path + '\\' + 'train_set.txt'):
        os.remove(path + '\\' + 'train_set.txt')

    for root_dir, sub_dirs, _ in os.walk(path):  # 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
        file_names=os.listdir(os.path.join(path,imgpath))
        file_names = list(filter(lambda x: x.endswith('.png'), file_names))  # 去掉列表中的非png格式的文件
        random.shuffle(file_names)
        for i in range(len(file_names)):
            if i < math.floor(0.8 * len(file_names)):
                txt_name = 'train_set.txt'
            elif i < math.floor(0.9 * len(file_names)):
                txt_name = 'val_set.txt'
            elif i < len(file_names):
                txt_name = 'test_set.txt'
            with open(os.path.join(path, txt_name), mode='a') as file:
                file.write(os.path.join(path, imgpath,file_names[i]) + '\n')
                # 为了以后好用，修改了这里，将' '改成了','，另外路径加了sub_dir

def ImgReadTxt(txt):
    with open(txt, 'r') as fh:
        imgs = []
        segs = []
        for line in fh:
            line1 = line.rstrip()  # 删除末尾空
            line2 = line1.replace('.png', '_P.png')
            line2 = line2.replace('images', 'labels')
            imgs.append(line1)
            segs.append(line2)
    return imgs,segs

def Cv2ImgOpen(fn,fn2,size=size0):

    image = cv2.imread(str(fn))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(fn2), 0)
    image = cv2.resize(image, (size,size))
    mask = cv2.resize(mask, (size,size))
    #image = image.reshape(3, size,size)
    #mask = mask.reshape(1, size,size)
    image=image.transpose(2,0,1)
    mask=np.expand_dims(mask,0)

    return image,mask

class RoadDataset(Dataset):
    # 构造函数设置默认参数
    def __init__(self, txt, transform=transforms.ToTensor()):
        imgs,segs=ImgReadTxt(txt)
        self.imgs = imgs
        self.segs=segs
        self.transform = transform

    def __getitem__(self, index):
        fn= self.imgs[index]
        fn2=self.segs[index]
        image,mask=Cv2ImgOpen(fn,fn2)
        return image, mask

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    datapath=r'E:\Desktop\unet\camvid'
    img='images'
    label='labels'
    batchsize=16
    transform = transforms.Compose([transforms.ToTensor()])

    txt = datapath + '\\' + 'train_set.txt'
    with open(txt, 'r') as fh:
        imgs = []
        segs = []
        for line in fh:
            '''
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            imgs=imgs.append(line)
            '''
            line1 = line.rstrip()  # 删除末尾空
            line2 = line1.replace('.png', '_P.png')
            line2 = line2.replace('images', 'labels')
            imgs.append(line1)
            segs.append(line2)
    fn = imgs[1]
    # segpath = fn.replace('.png', '_P.png')
    fn2 = segs[1]
    image = cv2.imread(str(fn))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(fn2), 0)

    mask=cv2.resize(mask,(240,240))
    print(mask.shape)
    print(mask)

    cv2.imshow('images', mask)
    cv2.imshow('im', image)
    cv2.waitKey(0)

'''
    DataDiv(datapath,img)

    transform = transforms.Compose([transforms.ToTensor()])

    data=RoadUnetDataset(datapath)

    train_loader = DataLoader(dataset=data, batch_size=batchsize, shuffle=True)
    print(data[0][0].shape)
    print(data[0][1].shape)
    #print(data[2][1])
    train_data=RoadDataset(txt=datapath + '\\' + 'train_set.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    print(train_data)
    print(train_loader)
'''
