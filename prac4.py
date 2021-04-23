import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

import argparse

class my_ResNet(nn.Module):
    def __init__(self):
        super(my_ResNet, self).__init__()
        self.start_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = block3x3(16, 16)

        self.layer2 = bottleneck3x3(16, 32, stride=1)
        self.layer2_res = bottleneck1x1(16, 32, stride=1)

        self.layer3 = block3x3(32, 32)

        self.layer4 = bottleneck3x3(32, 64, stride=2)
        self.layer4_res =bottleneck1x1(32, 64, stride=2)

        self.layer5 = block3x3(64, 64)

        self.layer6 = bottleneck3x3(64, 128, stride=2)
        self.layer6_res = bottleneck1x1(64, 128, stride=2)

        self.layer7 = block3x3(128, 128)

        self.layer8 = bottleneck3x3(128, 256)
        self.layer8_res = bottleneck1x1(128, 256)

        self.layer9 = block3x3(256, 256)

        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.start_layer(x)
        x = self.relu(self.layer1(x.clone()) + x)

        x = self.relu(self.layer2(x.clone()) + self.layer2_res(x))
        x = self.relu(self.layer3(x.clone()) + x)

        x = self.relu(self.layer4(x.clone()) + self.layer4_res(x))
        x = self.relu(self.layer5(x.clone()) + x)

        x = self.relu(self.layer6(x.clone()) + self.layer6_res(x))
        x = self.relu(self.layer7(x.clone()) + x)

        # x = self.relu(self.layer8(x.clone()) + self.layer8_res(x))
        # x = self.relu(self.layer9(x.clone()) + x)

        x = self.avg_pool(x)
        x = self.fc_layer(x)

        return x

def bottleneck3x3(in_channel, out_channel, stride=1):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
    )

def bottleneck1x1(in_channel, out_channel, stride=1):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
    )

def block3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
    )

def identity(out_channel):
    return nn.BatchNorm2d(out_channel)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 100),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm1d(100),
            nn.Linear(100, 10)
        )



        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = self.layer(x)
        x = self.fc_layer(x)

        return x

def compute_acc(dloader=None, model=None):
    correct = 0
    total = 0

    model.eval()

    for i, [imgs, labels] in enumerate(dloader):
        imgs = imgs.cuda()
        labels = labels.cuda()

        output = model(imgs)
        _, output_index = torch.max(output, 1)

        total += label.size(0)
        correct += (output_index == labels).sum().float()

    print(f'Accuracy of Test Data: {100 * correct/total}')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=10, type=int)

    args, fire_args = parser.parse_known_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs

    cifar_train = dsets.CIFAR10(root="CIFAR10/",
                                train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # data normalization
                                    # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                ]),
                                target_transform=None,
                                download=True)

    cifar_test  = dsets.CIFAR10(root="CIFAR10/",
                              train=False,
                              transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # data augmentation
                                    # data normalization
                                    # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                ]),
                              target_transform=None,
                              download=True)

    train_loader = DataLoader(dataset=cifar_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=20,
                              drop_last=True)
    test_loader = DataLoader(dataset=cifar_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=20,
                             drop_last=True)

    print(f'cifar_train 길이 : {len(cifar_train)}')
    print(f'cifar_test  길이 : {len(cifar_test)}')

    # 데이터 하나의 형태
    img, label = cifar_train.__getitem__(1)
    print(f'image data 형태 : {img.shape}')
    print(f'label           : {label}')

    # model = CNN().cuda()
    model = my_ResNet().cuda()
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    from tqdm import tqdm
    for epoch in range(epochs):
        model.train()

        for img, label in train_loader:
            img = img.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            output = model.forward(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

        print(f'epoch: {epoch:3d}  loss: {loss:.4f}')
        compute_acc(test_loader, model=model)


    torch.save(model, 'resnet_model1.pth')



