import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from vit_pytorch_robust import levit, vit, datasets, swin_t, patch_convnet
from vit_pytorch_robust.utils import rand_bbox

import argparse
from pathlib import Path
import numpy as np
import submitit
import omega
import logging


class Model(omega.Trainer):
    def initialize_train_loader(self):
        train_dataset = datasets.imagenet_train_dataset(
            Path("/datasets01/tinyimagenet/081318/")
            # Path("/datasets01/imagenet_full_size/061417/")
        )
        per_device_batch_size = self.args.batch_size // self.args.world_size
        if self.args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True, drop_last=False
            )
            self.train_sampler = train_sampler
            shuffle = None
        else:
            train_sampler = None
            shuffle = True
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=per_device_batch_size,
            num_workers=10,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=shuffle,
        )
        return train_loader

    def initialize_val_loader(self):
        val_dataset = datasets.imagenet_val_dataset(
            Path("/datasets01/tinyimagenet/081318/")
            # Path("/datasets01/imagenet_full_size/061417/")
        )
        per_device_batch_size = self.args.batch_size // self.args.world_size
        if self.args.world_size > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
            shuffle = None
        else:
            val_sampler = None
            shuffle = False
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=per_device_batch_size,
            num_workers=10,
            pin_memory=True,
            sampler=val_sampler,
            shuffle=shuffle,
        )
        return val_loader

    def initialize_modules(self):
        if self.args.architecture == "levit":
            self.model = levit.LeViT_128S(
                num_classes=self.args.num_classes, robust=self.args.robust
            )
            # self.model = SimpleViT(
            #     image_size=224,
            #     patch_size=args.ps,
            #     num_classes=200,
            #     dim=args.dim,
            #     depth=args.depth,
            #     heads=args.heads,
            #     mlp_dim=args.mlp_dim,
            #     robust=False,
            # ).cuda()
        elif self.args.architecture == "S60":
            self.model = patch_convnet.S60(
                num_classes=self.args.num_classes, robust=self.args.robust
            )
        elif self.args.architecture == "swin":
            self.model = swin_t(
                num_classes=self.args.num_classes, robust=self.args.robust
            )

    def initialize_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

    def initialize_scheduler(self):
        train_steps = len(self.train_loader)
        T1 = int(self.args.epochs * 0.1) * train_steps
        T2 = (self.args.epochs - int(self.args.epochs * 0.1)) * train_steps
        scheduler = SequentialLR(
            self.optimizer,
            [
                LinearLR(self.optimizer, 1e-3, 1, total_iters=T1),
                CosineAnnealingLR(
                    self.optimizer, T_max=T2, eta_min=self.args.learning_rate * 0.05
                ),
            ],
            milestones=[T1],
        )
        return scheduler

    def compute_loss(self):

        x = self.data[0].cuda(self.this_device, non_blocking=True)
        y = self.data[1].cuda(self.this_device, non_blocking=True)
        r = np.random.rand(1)
        if r < self.args.cutmix_prob:
            # generate mixed sample
            rand_index = torch.randperm(x.size()[0]).cuda(set_to_none=True)
            lam = np.random.beta(self.args.beta, self.args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2))

        preds = self.model(x)
        if r < self.args.cutmix_prob:
            loss = torch.nn.functional.cross_entropy(
                preds, y, label_smoothing=0.1
            ) * lam + torch.nn.functional.cross_entropy(
                preds, y[rand_index], label_smoothing=0.1
            ) * (
                1 - lam
            )
        else:
            loss = torch.nn.functional.cross_entropy(preds, y, label_smoothing=0.1)

        return loss

    def before_eval_epoch(self):
        super().before_eval_epoch()
        self.accu = 0
        self.counter = 0

    def eval_step(self):

        x = self.data[0].cuda(self.this_device, non_blocking=True)
        y = self.data[1].cuda(self.this_device, non_blocking=True)
        preds = self.model(x)
        accu = preds.argmax(1).eq(y).float().mean()
        torch.distributed.reduce(accu, dst=0)
        self.accu += accu.item()
        self.counter += 1

    def after_eval_epoch(self):
        super().after_eval_epoch()
        self.log_txt(
            "eval_accuracies",
            accus=(self.accu / self.counter) / self.args.world_size,
        )

    def after_train_step(self):
        self.scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--beta", default=1.0, type=float, help="hyperparameter beta")
    parser.add_argument(
        "--cutmix_prob", default=0.0, type=float, help="cutmix probability"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="swin",
        choices=["swin", "levit", "cait", "S60"],
    )
    parser.add_argument("--robust", action="store_true")

    omega.argparse.make_config(parser)

    args = parser.parse_args()
    # if not args.robust and args.architecture == "swin":
    #     args.learning_rate = 5e-4
    # elif args.robust and args.architecture == "swin":
    #     args.learning_rate = 5e-4
    args.weight_decay = 0.05
    args.grad_max_norm = 5.0
    args.epochs = 100
    args.sync_batchnorm = True
    args.eval_each_epoch = True
    args.batch_size = 512
    if args.architecture == "S60":
        args.batch_size = 256

    model = Model(args)
    executor = submitit.AutoExecutor(folder=model.args.folder)

    executor.update_parameters(
        timeout_min=1400,
        slurm_signal_delay_s=120,
        nodes=1,
        gpus_per_node=8,
        tasks_per_node=8,
        cpus_per_task=10,
        slurm_partition="learnlab",
        name="test",
        slurm_setup=[
            "module purge",
            "module load anaconda3/2020.11-nold",
            "source activate /private/home/rbalestriero/.conda/envs/ffcv",
        ],
    )
    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
