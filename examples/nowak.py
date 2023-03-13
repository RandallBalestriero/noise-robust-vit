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


class PartialReLU(torch.nn.ReLU):
    def forward(self, x):
        N = x.size(0)
        with torch.no_grad():
            mask = (
                x[: N // 2]
                .gt(0.0)
                .tile(2, *[1 for _ in range(x.ndim - 1)])
                .type(dtype=x.dtype, non_blocking=True)
            )
        return x * mask


def replace_modules(model):
    model = omega.utils.replace_module(model, torch.nn.ReLU, PartialReLU)

    def module_to_args_kwargs(module):
        return [], dict(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            process_group=module.process_group,
        )

    model = omega.utils.replace_module(
        model, torch.nn.SyncBatchNorm, PartialSyncBatchNorm, module_to_args_kwargs
    )
    return model


class Model(omega.Trainer):
    def initialize_train_loader(self):
        transforms = omega.transforms.imagenet_train_dataset(
            strength=self.args.strength
        )
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
        import torchvision

        model = torchvision.models.__dict__[self.args.architecture]()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = replace_modules(model)

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
        y = self.data[1].cuda(self.this_device, non_blocking=True)
        N = x.size(0)
        eps = torch.randn_like(x) * self.args.noise_std
        preds = self.model(torch.cat([x + eps, x], 0))
        if self.args.improved:
            loss = torch.nn.functional.cross_entropy(preds[N:], y, label_smoothing=0.1)
        else:
            loss = torch.nn.functional.cross_entropy(preds[:N], y, label_smoothing=0.1)
        return loss

    def before_eval_epoch(self):
        super().before_eval_epoch()
        self.accu = 0
        self.counter = 0

    def eval_step(self):
        x = self.data[0].cuda(self.this_device, non_blocking=True)
        y = self.data[1].cuda(self.this_device, non_blocking=True)
        preds = self.model(torch.cat([x, x], 0))[: x.size(0)]
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
    parser.add_argument("--improved", action="store_true")
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
    )
    parser.add_argument("--strength", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--noise-std", type=float, default=0.1)
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
    args.epochs = 100
    args.eval_each_epoch = True
    args.batch_size = 128

    model = Model(args)
    executor = submitit.AutoExecutor(folder=model.args.folder)

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
