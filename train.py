import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.optim import lr_scheduler
import get_data

from tqdm import tqdm

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

    print(f'Accuracy of Test Data: {100 * correct/total}')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 100),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm1d(100),
            nn.Linear(100, 10)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            if isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = self.layer(x)
        x = self.fc_layer(x)

        return x

def training(batch_size, epochs, learning_rate):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN().to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # (6) Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # (7) learning rate decay
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    train_loader, test_loader = get_data.get_cifar10_dataloader(batch_size=batch_size)

    for epoch in range(epochs):
        model = model.train()

        for img, label in tqdm(train_loader):
            x = Variable(img).to(device)
            y = Variable(label).to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

        print(f'epoch: {epoch:3d}  loss: {loss:4f}')

        compute_acc(test_loader, model=model)

if __name__ == '__main__':
    training(64, 1, 1e-3)