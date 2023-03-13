import torch
import argparse
import submitit
import omega
from pathlib import Path


class Model(omega.Trainer):
    def initialize_train_loader(self):
        transforms = omega.transforms.imagenet_train_dataset(
            strength=self.args.strength
        )
        train_dataset, self.num_classes = omega.dataset.get_dataset(
            self.args.dataset_path, transforms, split="train"
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
        transforms = omega.transforms.imagenet_val_dataset()
        val_dataset, _ = omega.dataset.get_dataset(
            self.args.dataset_path, transforms, split="test"
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
        import torchvision

        self.model = torchvision.models.__dict__[self.args.architecture]()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)

    def compute_loss(self):
        x = self.data[0].cuda(self.this_device, non_blocking=True)
        y = self.data[1].cuda(self.this_device, non_blocking=True)
        preds = self.model(x)
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
    parser.add_argument("--strength", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--dataset-path",
        default="/datasets01/tinyimagenet/081318/",
        type=Path,
        choices=[
            Path("/datasets01/tinyimagenet/081318/"),
            Path("/datasets01/imagenet_full_size/061417/"),
            Path("/datasets01/Places365/041019/"),
            Path("/datasets01/inaturalist/111621/"),
            Path("/private/home/rbalestriero/DATASETS/OxfordIIITPet/"),
            Path("/private/home/rbalestriero/DATASETS/Omniglot/"),
            Path("/private/home/rbalestriero/DATASETS/Food101/"),
            Path("/private/home/rbalestriero/DATASETS/Flowers102/"),
            Path("/private/home/rbalestriero/DATASETS/FGVCAircraft/"),
            Path("/private/home/rbalestriero/DATASETS/StanfordCars/"),
            Path("/private/home/rbalestriero/DATASETS/Country211/"),
            Path("/private/home/rbalestriero/DATASETS/DTD/"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR10/"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR100/"),
            Path("/datasets01/places205/121517/pytorch/"),
        ],
    )
    omega.argparse.make_config(parser)

    args = parser.parse_args()
    args.grad_max_norm = 5.0
    args.eval_each_epoch = True
    if args.architecture in ["resnet101"]:
        args.gpus_per_node = 4
    else:
        args.gpus_per_node = 2

    model = Model(args)
    executor = submitit.AutoExecutor(folder=model.args.folder)

    executor.update_parameters(
        timeout_min=1400,
        slurm_signal_delay_s=120,
        nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        tasks_per_node=args.gpus_per_node,
        cpus_per_task=10,
        slurm_partition="devlab",
        name="baseline",
        slurm_setup=[
            "module purge",
            "module load anaconda3/2020.11-nold",
            "source activate /private/home/rbalestriero/.conda/envs/ffcv",
        ],
    )
    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
