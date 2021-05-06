#%%

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

from tqdm import tqdm


batch_size = 1024
learning_rate = 1e-2
epochs = 2000

#%%
cifar_train = dsets.CIFAR10(root="CIFAR10/",
                            train=True,
                            transform=transforms.Compose([
                                # data augmentation
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                # data normalization
                                transforms.ToTensor(),
                                transforms.Normalize(
                                  mean= (0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                  std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                            ]),
                            target_transform=None,
                            download=True)

cifar_test  = dsets.CIFAR10(root="CIFAR10/",
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(
                                mean= (0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                          ]),
                          target_transform=None,
                          download=True)

#%%
print(f'cifar_train 길이 : {len(cifar_train)}')
print(f'cifar_test  길이 : {len(cifar_test)}')

img, label = cifar_train.__getitem__(1)
print(f'image data 형태 : {img.shape}')
print(f'label           : {label}')

plt.title(f'title : {label}')
print(img)
plt.imshow(img, interpolation='bicubic')
plt.show()

#%%
def compute_acc(dloader=None, model=None):
    model.eval()
    correct = 0
    total = 0

    for i, [imgs, labels] in enumerate(dloader):
        img = Variable(imgs, requires_grad=False).cuda()
        label = Variable(labels).cuda()

        output = model(img)
        _, output_index = torch.max(output, 1)

        total += label.size(0)
        correct += (output_index == label).sum().float()

    return 100 * correct/total

train_loader = DataLoader(dataset=cifar_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True)
test_loader  = DataLoader(dataset=cifar_test,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8,
                          drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ELU(),
            #nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(num_features=16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ELU(),
            #nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64 * 4 * 4),
            nn.ELU(),
            nn.BatchNorm1d(64 * 4 * 4),
            nn.Dropout2d(p=0.5),

            nn.Linear(64 * 4 * 4, 64 * 2 * 2),
            nn.ELU(),
            nn.BatchNorm1d(64 * 2 * 2),
            nn.Dropout2d(p=0.5),

            nn.Linear(64 * 2 * 2, 10)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.layer(x)
        x = self.fc_layer(x)

        return x

# from torch.utils.tensorboard import SummaryWriter

def training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4)

    # net_name = './weights/best-201710860.pkl'
    net_name = '<net_name>'
    max_acc = 0
    for epoch in tqdm(range(epochs)):
        model = model.train()

        for img, label in train_loader:
            x = Variable(img).to(device)
            y = Variable(label).to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()
            scheduler.step()

        acc = compute_acc(test_loader, model)

        if acc > max_acc:
            print(acc)
            max_acc = acc
            torch.save(model, net_name)
#%%
training()





