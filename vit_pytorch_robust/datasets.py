import numpy as np
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torch


def imagenet(path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size = 224

    train_transform = T.Compose(
        [
            T.RandomResizedCrop((size, size)),
            T.RandomApply(
                torch.nn.ModuleList([T.ColorJitter(0.8, 0.8, 0.8, 0.2)]), p=0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize(int(size * 1.14)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = ImageFolder(path / "train", transform=train_transform)
    val_dataset = ImageFolder(path / "val", transform=val_transform)
    return train_dataset, val_dataset
