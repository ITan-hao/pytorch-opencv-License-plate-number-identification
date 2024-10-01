'''
功能：训练一个车牌字符识别模型
备注：训练出来的模型文件为 Model.pth 可加载这个模型使用
    目前的准确率在 85% 以后可以选择较好的模型提高准确率
'''
import torch
# import torchvision

from torch.utils.data import Dataset, DataLoader
import os
import time
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import math
import cv2

import torchvision.datasets as datasets
import torchvision.transforms as transforms



LR = 0.00005     # 设置学习率
EPOCH_NUM = 30


def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)


# 初始化根目录
train_path  = 'data/train'
test_path   = 'data/train'



# 定义读取文件的格式
# def default_loader(path):
#     return Image.open(path).convert('RGB')

transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 40)), # 原本就是 32x40 不需要修改尺寸
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
traindataset = datasets.ImageFolder(train_path,transform)
testdataset = datasets.ImageFolder(test_path,transform)


# 数据集类
class MyDataSet(Dataset):
    def __init__(self, data_path:str, transform):  # 传入训练样本路径
        super(MyDataSet, self).__init__()
        datas =open(data_path,'r').readlines()
        self.images = []
        self.labels = []
        self.transform = transform
        for data in datas:
            item = data.strip().split(' ')
            self.images.append(item[0])
            self.labels.append(item[1])


    def __getitem__(self, item):
        image = torch.as_tensor(self.images[item], dtype=torch.int64)
        label = torch.as_tensor(self.labels[item], dtype=torch.int64)
        return transform(image), label

    def __len__(self)->int:
        return len(self.path_list)


train_ds = MyDataSet(train_path,transform)
test_data = MyDataSet(test_path,transform)
# for i, item in enumerate(tqdm(train_ds)):
#     print(item)
#     break

# 数据加载
new_train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
new_test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

# for i, item in enumerate(new_train_loader):
#     print(item[0].shape)
#     break
#
# img_PIL_Tensor = train_ds[1][0]
# new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
# plt.imshow(new_img_PIL)
# plt.show()


# 设置训练类
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpooling = torch.nn.MaxPool2d(2)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.globalavgpool = torch.nn.AvgPool2d((8, 10))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.dropout50 = torch.nn.Dropout(0.5)
        self.dropout10 = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(256, 40)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpooling(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.maxpooling(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(epoch, loss_list):
    running_loss = 0.0
    for batch_idx, data in enumerate(new_train_loader, 0):
        inputs, target = data[0], data[1]
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs.shape, target.shape)
        # print(outputs, target)
        # break
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print('[%d, %5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

    return loss_list

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(new_test_loader, 0):
            inputs, target = data[0], data[1]
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, dim=1)

            total += target.size(0)
            correct += (prediction == target).sum().item()
    print('Accuracy on test set: (%d/%d)%d %%' % (correct, total, 100 * correct / total))


if __name__ == '__main__':
    start = time.time()
    loss_list = []
    for epoch in range(EPOCH_NUM):
        train(epoch, loss_list)
        test()
    torch.save(model.state_dict(), "Model.pth")     # 训练完成 保存训练好的模型

    x_ori = []
    for i in range(len(loss_list)):
        x_ori.append(i)
    plt.title("Graph")
    plt.plot(x_ori, loss_list)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()
