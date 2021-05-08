import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size):
    '''
    return cifar10 dataset wrapping in dataloader with batsh size

    :param batch_size: batch size of dataloader
    :return: (trainset dataloaderm, testset dataloader)
    '''

    cifar_train = dsets.CIFAR10(root="CIFAR10/",
                                train=True,
                                transform=transforms.Compose([
                                    # data augmentation
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=30),

                                    # data normalization
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                        std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])
                                    ]),
                                target_transform=None,
                                download=True)

    cifar_test  = dsets.CIFAR10(root="CIFAR10/",
                              train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                      std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])
                              ]),
                              target_transform=None,
                              download=True)

    train_loader = DataLoader(dataset=cifar_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6,
                              drop_last=True)
    test_loader  = DataLoader(dataset=cifar_test,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=6,
                              drop_last=True)

    return train_loader, test_loader