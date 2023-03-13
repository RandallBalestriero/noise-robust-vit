import torch
import argparse
import submitit
import omega
import numpy as np
from pathlib import Path

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Model(omega.Trainer):
    def initialize_train_loader(self):
        name = self.args.train_dataset_path.parent.name.lower()
        self.num_classes, self.image_size = omega.dataset.NAME_TO_CLASS[name]
        if "vit" in self.args.architecture or "swin" in self.args.architecture:
            self.image_size = 224
        elif self.args.architecture == "alexnet":
            self.image_size = max(self.image_size, 64)

        if (
            "CIFAR" in self.args.train_dataset_path.absolute().as_posix()
            or "TINY" in self.args.train_dataset_path.absolute().as_posix()
        ):
            ratio = 1.0
        else:
            ratio = 224 / 256
        pipes = omega.transforms.ffcv_imagenet_train_dataset(
            device=self.this_device,
            dtype=torch.float16 if self.args.float16 else torch.float32,
            strength=self.args.strength,
            size=self.image_size,
            ratio=ratio,
        )

        train_loader = omega.ffcv.train_reader(
            path=self.args.train_dataset_path,
            pipelines={"image": pipes[0], "label": pipes[1]},
            batch_size=self.args.batch_size,
            world_size=self.args.world_size,
        )

        return train_loader

    def initialize_val_loader(self):
        if (
            "CIFAR" in self.args.train_dataset_path.absolute().as_posix()
            or "TINY" in self.args.train_dataset_path.absolute().as_posix()
        ):
            ratio = 1.0
        else:
            ratio = 224 / 256
        pipes = omega.transforms.ffcv_imagenet_val_dataset(
            device=self.this_device,
            dtype=torch.float16 if self.args.float16 else torch.float32,
            size=self.image_size,
            ratio=ratio,
        )

        val_loader = omega.ffcv.train_reader(
            path=self.args.val_dataset_path,
            pipelines={"image": pipes[0], "label": pipes[1]},
            batch_size=self.args.batch_size,
            world_size=self.args.world_size,
        )
        return val_loader

    def initialize_modules(self):
        import torchmetrics
        from torchmetrics.classification import MulticlassAccuracy

        self.valid_accuracy = MulticlassAccuracy(self.num_classes, top_k=1).to(
            self.this_device
        )
        self.valid_accuracy5 = MulticlassAccuracy(self.num_classes, top_k=5).to(
            self.this_device
        )
        self.train_loss = torchmetrics.MeanMetric().to(self.this_device)

        is_cifar = "CIFAR" in self.args.train_dataset_path.absolute().as_posix()

        model, fan_in = omega.utils.load_without_classifier(self.args.architecture)

        if is_cifar and "resnet" in self.args.architecture:
            model.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            model.maxpool = torch.nn.Identity()

        self.projector = torch.nn.Linear(
            fan_in,
            self.num_classes,
        )
        self.classifier = torch.nn.Linear(fan_in, self.num_classes)
        self.model = model

    def compute_loss(self):
        x = self.data[0]
        labels = self.data[1][:, 0]
        preds = self.model(x)

        # online probe
        preds_true = self.classifier(preds.detach())
        true_loss = torch.nn.functional.cross_entropy(preds_true, labels)

        G = labels[:, None].eq(labels)
        Z = self.projector(preds)
        C = torch.cov(Z.t())
        I = torch.eye(C.size(0), dtype=C.dtype, device=C.device)
        VC_loss = (C - I).square().mean()
        i, j = G.nonzero(as_tuple=True)
        inv_loss = (Z[i] - Z[j]).square().mean()
        other_loss = VC_loss + self.args.temperature * inv_loss

        self.train_loss.update(other_loss)
        return other_loss + true_loss

    def eval_step(self):
        x, y = self.data
        preds = self.classifier(self.model(x))
        self.valid_accuracy.update(preds, y)
        self.valid_accuracy5.update(preds, y)

    def after_eval_epoch(self):
        super().after_eval_epoch()
        accu = self.valid_accuracy.compute().item()
        self.log_txt(
            "eval_accuracies",
            accus=accu,
            accus5=self.valid_accuracy5.compute().item(),
            train_loss=self.train_loss.compute().item(),
        )
        # Reset metric states after each epoch
        self.valid_accuracy.reset()
        self.valid_accuracy5.reset()
        self.train_loss.reset()

    def initialize_scheduler(self):
        if self.args.epochs > 100:
            return super().initialize_scheduler()
        N = len(self.train_loader)
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                int(self.args.epochs * 0.5) * N,
                int(self.args.epochs * 0.75) * N,
            ],
            gamma=0.1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--strength", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--train-dataset-path",
        default="/datasets01/tinyimagenet/081318/",
        type=Path,
        choices=[
            Path("/datasets01/Places365/041019/"),
            Path("/datasets01/inaturalist/111621/"),
            Path("/private/home/rbalestriero/DATASETS/OxfordIIITPet/"),
            Path("/private/home/rbalestriero/DATASETS/FGVCAircraft/train_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR10/train_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/IMAGENET/train_500_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/IMAGENET100/train_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CUB_200_2011/train_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/DTD/train_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/OxfordIIITPet/train_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/Food101/train_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/StanfordCars/train_jpg.ffcv"),
            Path(
                "/private/home/rbalestriero/DATASETS/INATURALIST/train_500_0.50_90_0.ffcv"
            ),
            Path("/private/home/rbalestriero/DATASETS/Flowers102/train_jpg.ffcv"),
        ],
    )
    parser.add_argument(
        "--val-dataset-path",
        default="/datasets01/tinyimagenet/081318/",
        type=Path,
        choices=[
            Path("/datasets01/Places365/041019/"),
            Path("/datasets01/inaturalist/111621/"),
            Path("/private/home/rbalestriero/DATASETS/FGVCAircraft/test_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR10/val_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR100/val_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/IMAGENET/val_500_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/IMAGENET100/val_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CUB_200_2011/test_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/DTD/test_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/OxfordIIITPet/test_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/Food101/test_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/StanfordCars/test_jpg.ffcv"),
            Path(
                "/private/home/rbalestriero/DATASETS/INATURALIST/val_500_0.50_90_0.ffcv"
            ),
            Path("/private/home/rbalestriero/DATASETS/Flowers102/test_jpg.ffcv"),
        ],
    )
    omega.argparse.make_config(parser)

    args = parser.parse_args()

    model = Model(args)
    if args.local:
        model()
        import sys

        sys.exit()
    executor = submitit.AutoExecutor(
        folder=model.args.folder, slurm_max_num_timeout=450
    )

    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_signal_delay_s=120,
        nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        tasks_per_node=args.gpus_per_node,
        cpus_per_task=10,
        slurm_partition=args.slurm_partition,
        name=args.process_name,
        slurm_setup=[
            "module purge",
            "module load anaconda3/2020.11-nold",
            "source activate /private/home/rbalestriero/.conda/envs/ffcv",
        ],
    )

    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
