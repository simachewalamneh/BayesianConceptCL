import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

SPLIT_MNIST_TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def get_split_mnist_tasks(root: str = "./data", batch_size: int = 128):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    train_loaders, test_loaders = [], []
    for classes in SPLIT_MNIST_TASKS:
        train_idx = [i for i, t in enumerate(train_full.targets) if t.item() in classes]
        test_idx = [i for i, t in enumerate(test_full.targets) if t.item() in classes]
        train_loaders.append(DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(test_full, test_idx), batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders
