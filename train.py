import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.optim import lr_scheduler
import get_data

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
            nn.ELU(),
            # nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(num_features=16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            # nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64 * 4 * 4),
            nn.ELU(),
            nn.Dropout2d(p=0.5),
            nn.BatchNorm1d(64 * 4 * 4),

            nn.Linear(64 * 4 * 4, 64 * 4 * 4),
            nn.ELU(),
            nn.Dropout2d(p=0.5),
            nn.BatchNorm1d(64 * 4 * 4),

            nn.Linear(64 * 4 * 4, 10),
            nn.BatchNorm1d(10),
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

    comment = 'fc-extened-model-2000-epoch-lr=1e-3-rotation30-horizontalflip'

    max_acc = 0
    with SummaryWriter(comment=comment) as writer:
        writer.add_text(tag='model arch', text_string=str(model))

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

            acc = compute_acc(test_loader, model)
            writer.add_scalar('training loss', loss, epoch + 1)
            writer.add_scalar('training accuarcy', acc, epoch + 1)

            if acc > max_acc:
                torch.save(model, './weights/best-model-fc-extended.pth')
                max_acc = acc

    print(f'best acc = {max_acc}')

def fine_tune(batch_size, epochs, learning_rate):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('./weights/best-model-2000-epoch2.pth')
    model.train()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # (6) Adam optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adadelta(params=model.parameters())

    # (7) learning rate decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_loader, test_loader = get_data.get_cifar10_dataloader(batch_size=batch_size)

    comment = 'fc-8x8-2x2-1x1-best-model-2000-epoch2-finetune-lr=1e-5-rotation30'

    max_acc = 0
    with SummaryWriter(comment=comment) as writer:
        writer.add_text(tag='model arch', text_string= str(model))

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

            acc = compute_acc(test_loader, model)
            writer.add_scalar('training loss', loss, epoch+1)
            writer.add_scalar('training accuarcy', acc, epoch+1)

            if acc > max_acc:
                torch.save(model, './weights/best-model-2000-epoch2-finetuned.pth')
                max_acc = acc

    print(f'best acc = {max_acc}')

if __name__ == '__main__':
    torch.manual_seed(777)

    training(512, 3000, 1e-3)
    # fine_tune(512, 1000, 1e-3)

    #model = torch.load('./weights/best-model-2000-epoch3.pth').to('cuda')
    #_, test_loader = get_data.get_cifar10_dataloader(512)
    #acc = compute_acc(test_loader, model)
    #print(acc)