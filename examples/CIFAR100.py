import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from vit_pytorch_robust import ViT, datasets, SimpleViT, levit, swin_t
from vit_pytorch_robust.utils import (
    seed_everything,
    main,
    rand_bbox,
    save_all,
    load_all,
    get_open_port,
)

import argparse
import sys
import time
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os


def main_worker(gpu, args):

    seed_everything(args.seed)

    args.rank += gpu

    if args.world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_stats_file = open(
            args.checkpoint_dir / "train_stats.txt", "a", buffering=1
        )
        val_stats_file = open(args.checkpoint_dir / "val_stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=train_stats_file)
        print(" ".join(sys.argv), file=val_stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # Dataset loaders
    train_dataset, val_dataset = datasets.imagenet(
        Path("/datasets01/tinyimagenet/081318/")
        # Path("/datasets01/imagenet_full_size/061417/")
    )

    per_device_batch_size = args.batch_size // args.world_size

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        shuffle = None
    else:
        val_sampler = None
        shuffle = True
        train_sampler = None
    assert args.batch_size % args.world_size == 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        num_workers=10,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=shuffle,
    )
    train_steps = len(train_loader)

    if args.world_size > 1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        shuffle = None
    else:
        val_sampler = None
        shuffle = False
    assert args.batch_size % args.world_size == 0
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        num_workers=10,
        pin_memory=True,
        sampler=val_sampler,
        shuffle=shuffle,
    )

    # models
    # model = SimpleViT(
    #     image_size=224,
    #     patch_size=args.ps,
    #     num_classes=200,
    #     dim=args.dim,
    #     depth=args.depth,
    #     heads=args.heads,
    #     mlp_dim=args.mlp_dim,
    #     robust=False,
    # ).cuda()
    model = swin_t(num_classes=200).cuda()
    # model = levit.LeViT(
    #     image_size=224,
    #     num_classes=200,
    #     stages=3,  # number of stages
    #     dim=(256, 384, 512),  # dimensions at each stage
    #     depth=args.depth,  # transformer of depth 4 at each stage
    #     heads=(4, 6, 8),  # heads at each stage
    #     mlp_mult=2,
    #     dropout=0.1,
    # ).cuda()

    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1.0,
                total_iters=int(args.epochs * 0.1) * train_steps,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=(args.epochs - int(args.epochs * 0.1)) * train_steps,
                eta_min=1e-6,
            ),
        ],
        milestones=[3 * train_steps],
        verbose=True,
    )

    # automatically resume from checkpoint if it exists
    path = args.checkpoint_dir / "checkpoint.pth"
    if path.is_file():
        start_epoch = load_all(model, optimizer, scheduler, path)
    else:
        start_epoch = 0

    start_time = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == 16)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)

        for step, (x, y) in tqdm(
            enumerate(train_loader, start=epoch * train_steps),
            total=train_steps,
            desc=f"Training: {epoch=}",
            disable=args.rank > 0,
        ):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            r = np.random.rand(1)
            if r < args.cutmix_prob:
                # generate mixed sample
                rand_index = torch.randperm(x.size()[0]).cuda(set_to_none=True)
                lam = np.random.beta(args.beta, args.beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2))
            with torch.cuda.amp.autocast(enabled=args.precision == 16):
                preds = model(x)
                if r < args.cutmix_prob:
                    loss = torch.nn.functional.cross_entropy(
                        preds, y, label_smoothing=0.001
                    ) * lam + torch.nn.functional.cross_entropy(
                        preds, y[rand_index], label_smoothing=0.001
                    ) * (
                        1 - lam
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        preds, y, label_smoothing=0.001
                    )
            scaler.scale(loss).backward()  # to create scaled gradients
            scaler.step(optimizer)  # Unscales gradients and calls
            scaler.update()  # Updates the scale for next iteration
            scheduler.step()

            if step % args.log_freq == 0 and args.rank == 0:
                with torch.no_grad():
                    acc = (preds.argmax(dim=1) == y).float().mean().item()
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr=optimizer.param_groups[0]["lr"],
                        loss=loss.item(),
                        acc=100 * acc,
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats, indent=4))
                    print(json.dumps(stats), file=train_stats_file)

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for x, y in tqdm(val_loader, desc=f"Val: {epoch=}", disable=args.rank > 0):
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.precision == 16):

                    preds = model(x)

                val_loss = torch.nn.functional.cross_entropy(preds, y)
                acc = (preds.argmax(dim=1) == y).float().mean()
                if args.world_size > 1:
                    op = torch.distributed.ReduceOp.SUM
                    torch.distributed.all_reduce(val_loss, op=op)
                    torch.distributed.all_reduce(acc, op=op)
                epoch_val_accuracy += acc.item() / args.world_size
                epoch_val_loss += val_loss.item() / args.world_size
            if args.rank == 0:
                stats = dict(
                    epoch=epoch,
                    epoch_loss=epoch_val_loss / len(val_loader),
                    epoch_acc=100 * epoch_val_accuracy / len(val_loader),
                )
                print(json.dumps(stats, indent=4))
                print(json.dumps(stats), file=val_stats_file)

        if args.rank == 0:
            save_all(epoch, model, optimizer, scheduler, path)
    if args.rank == 0:
        # save final model
        path = args.checkpoint_dir / "final.pth"
        torch.save(model.module.backbone.state_dict(), path)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument(
        "--epochs",
        "-e",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        default=0.003,
        type=float,
        metavar="LR",
        dest="lr",
        help="base learning rate for weights",
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        default=0.03,
        type=float,
        metavar="W",
        dest="wd",
        help="weight decay",
    )
    parser.add_argument(
        "--log-freq", default=30, type=int, metavar="N", help="print/log frequency"
    )
    parser.add_argument(
        "--checkpoint-dir",
        "-cd",
        default="./checkpoint/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )

    parser.add_argument("--beta", default=1.0, type=float, help="hyperparameter beta")
    parser.add_argument(
        "--cutmix_prob", default=0.0, type=float, help="cutmix probability"
    )
    parser.add_argument("--patch_size", "-ps", type=int, default=32, dest="ps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])

    args = parser.parse_args()
    setattr(args, "port", get_open_port())
    main(args, main_worker)
