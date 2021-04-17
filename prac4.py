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
                init.kaiming_normal(m.weight.data)
                # m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight.data)
                # m.bias.fill_(0)

    def forward(self, x):
        x = self.layer(x)
        x = self.fc_layer(x)

        return x

def compute_acc(dloader=None, model=None):
    correct = 0
    total = 0

    for i, [imgs, labels] in enumerate(dloader):
        img = Variable(imgs, requires_grad=False).cuda()
        label = Variable(labels).cuda()

        output = model(img)
        _, output_index = torch.max(output, 1)

        total += label.size(0)
        correct += (output_index == label).sum().float()

    print(f'Accuracy of Test Data: {100 * correct/total}')

if __name__ == '__main__' :
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10

    cifar_train = dsets.CIFAR10(root="CIFAR10/",
                                train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # data augmentation
                                    transforms.Scale(36),
                                    transforms.CenterCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(90),
                                    # data normalization
                                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                ]),
                                target_transform=None,
                                download=True)

    cifar_test  = dsets.CIFAR10(root="CIFAR10/",
                              train=False,
                              transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # data augmentation
                                    # test set에서는 data augmentation을 하지 않는다
                                    # transforms.Scale(36),
                                    # transforms.CenterCrop(32),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.Lambda(lambda x: x.rotate(90)),
                                    # data normalization
                                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                ]),
                              target_transform=None,
                              download=True)

    train_loader = DataLoader(dataset=cifar_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)
    test_loader = DataLoader(dataset=cifar_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             drop_last=True)

    print(f'cifar_train 길이 : {len(cifar_train)}')
    print(f'cifar_test  길이 : {len(cifar_test)}')

    # 데이터 하나의 형태
    img, label = cifar_train.__getitem__(1)
    print(f'image data 형태 : {img.shape}')
    print(f'label           : {label}')

    model = CNN().cuda()
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from tqdm import tqdm
    for epoch in range(epochs):
        for img, label in tqdm(train_loader):
            x = Variable(img).cuda()
            y = Variable(label).cuda()

            optimizer.zero_grad()
    
            output = model.forward(x)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'epoch: {epoch:3d}  loss: {loss:4f}')

    compute_acc(test_loader, model=model)
