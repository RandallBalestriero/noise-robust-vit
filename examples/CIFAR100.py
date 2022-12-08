import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torchvision.datasets import ImageFolder

from vit_pytorch_robust import ViT
from vit_pytorch_robust.utils import seed_everything

import argparse


def main(args):

    seed_everything(args.seed)

    model = ViT(
        image_size=224,
        patch_size=32,
        num_classes=200,
        dim=1024,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        robust=True,
    ).to(args.device)

    # data loader
    # the magic normalization parameters come from the example
    transform_mean = np.array([0.485, 0.456, 0.406])
    transform_std = np.array([0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std),
        ]
    )
    train_dataset = ImageFolder(
        "/datasets01/tinyimagenet/081318/train",
        transform=train_transform,
    )
    val_dataset = ImageFolder(
        "/datasets01/tinyimagenet/081318/val",
        transform=val_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=3e-5
    )
    # scheduler
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1.0,
                total_iters=3 * len(train_loader),
            ),
            CosineAnnealingLR(optimizer, T_max=(args.epochs - 3) * len(train_loader)),
        ],
        milestones=[3 * len(train_loader)],
        verbose=True,
    )

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()

        for data, label in tqdm(train_loader):
            data = data.to(args.device)
            label = label.to(args.device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(args.device)
                label = label.to(args.device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=float, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)
