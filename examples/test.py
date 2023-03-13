import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torchvision.datasets import ImageFolder

import argparse
import submitit
import omega
import pathlib


class PartialSyncBatchNorm(torch.nn.SyncBatchNorm):
    def forward(self, x):
        N = x.size(0)
        P = torch.distributed.get_world_size()

        first, second = x.split(N // 2)

        # first part is original
        first_out = super().forward(first)

        # second part
        sum = first.mean((0, 2, 3), keepdims=True) / P
        ssum = first.square().mean((0, 2, 3), keepdims=True) / P
        torch.distributed.all_reduce(sum)
        torch.distributed.all_reduce(ssum)

        std = torch.sqrt(ssum - sum.square() + self.eps)
        second_out = self.weight.view_as(std) * (
            second - sum
        ) / std + self.bias.view_as(std)
        return torch.cat([first_out, second_out], 0)


class Model(omega.Trainer):
    def initialize_train_loader(self):
        transforms = omega.transforms.imagenet_train_dataset()
        train_dataset = train_dataset = ImageFolder(
            self.args.dataset_path / "train", transform=transforms
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
        transforms = omega.transforms.imagenet_train_dataset()
        val_dataset = ImageFolder(self.args.dataset_path / "val", transform=transforms)
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
        self.model = PartialSyncBatchNorm(3)

    def initialize_optimizer(self):
        return optim.AdamW(
            self.parameters(),
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
        preds = self.model(torch.cat([x, x], 0))
        print(x.shape, preds.shape)
        print(preds[:, :, 0, :5])
        asfd


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument(
        "--dataset-path",
        default="/datasets01/tinyimagenet/081318/",
        type=pathlib.Path,
        choices=[
            "/datasets01/tinyimagenet/081318/",
            "/datasets01/imagenet_full_size/061417/",
        ],
    )
    omega.argparse.make_config(parser)

    args = parser.parse_args()
    args.learning_rate = 5e-4
    args.weight_decay = 0.05
    args.grad_max_norm = 5.0
    args.batch_size = 4

    model = Model(args)
    executor = submitit.AutoExecutor(folder=model.args.folder, cluster="local")

    executor.update_parameters(
        timeout_min=1400,
        slurm_signal_delay_s=120,
        nodes=1,
        gpus_per_node=2,
        tasks_per_node=2,
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
