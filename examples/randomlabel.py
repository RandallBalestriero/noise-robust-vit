import torch
import torch.optim as optim
from torch.utils.data import random_split, Subset

from torchvision.datasets import ImageFolder

import argparse
import submitit
import omega
from pathlib import Path
from numpy import random

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


class MyReLU(torch.nn.ReLU):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p > 0:
            x = torch.nn.functional.dropout(x, p=self.p, training=self.training)
        x = torch.nn.functional.relu(x)
        return x


class MyDataset(ImageFolder):
    def __init__(self, data_source, percentage=1, stratify=False, seed=None):
        if percentage < 1 and not stratify:
            N = int(percentage * len(data_source))
            splits = [N, len(data_source) - N]
            generator = torch.Generator().manual_seed(seed)
            self.data_source = random_split(data_source, splits, generator=generator)[0]
        elif percentage < 1 and stratify:
            counts = {}
            for i, (_, y) in enumerate(data_source):
                if y not in counts:
                    counts[y] = []
                counts[y].append(i)
            generator = random.RandomState(seed)
            indices = []
            for values in counts.values():
                kwargs = dict(size=int(len(values) * percentage), replace=False)
                indices.extend(generator.choice(range(len(values)), **kwargs))
            self.data_source = Subset(data_source, indices)
        else:
            self.data_source = data_source

    def __getitem__(self, index):
        data, target = self.data_source[index]
        return data, target, index

    def __len__(self):
        return len(self.data_source)


class Model(omega.Trainer):
    def initialize_train_loader(self):
        name = self.args.train_dataset_path.parent.name.lower()
        self.num_classes, self.image_size = omega.dataset.NAME_TO_CLASS[name]

        pipes = omega.transforms.ffcv_imagenet_train_dataset(
            device=self.this_device,
            dtype=torch.float16 if self.args.float16 else torch.float32,
            strength=self.args.strength,
            size=self.image_size,
        )

        train_loader = omega.ffcv.train_reader(
            path=self.args.train_dataset_path,
            pipelines={"image": pipes[0], "label": pipes[1]},
            batch_size=self.args.batch_size,
            world_size=self.args.world_size,
        )
        self.train_samples = train_loader.reader.num_samples
        # transforms = omega.transforms.imagenet_train_dataset(
        #     strength=self.args.strength
        # )
        # data_source, self.num_classes = omega.dataset.get_dataset(
        #     self.args.dataset_path, transforms, split="train"
        # )
        # train_dataset = MyDataset(data_source)
        # self.train_samples = len(train_dataset)

        # per_device_batch_size = self.args.batch_size // self.args.world_size
        # if self.args.world_size > 1:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         train_dataset, shuffle=True, drop_last=False
        #     )
        #     self.train_sampler = train_sampler
        #     shuffle = None
        # else:
        #     train_sampler = None
        #     shuffle = True
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=per_device_batch_size,
        #     num_workers=10,
        #     pin_memory=True,
        #     sampler=train_sampler,
        #     shuffle=shuffle,
        # )
        return train_loader

    def initialize_val_loader(self):

        pipes = omega.transforms.ffcv_imagenet_val_dataset(
            device=self.this_device,
            dtype=torch.float16 if self.args.float16 else torch.float32,
            size=self.image_size,
        )

        val_loader = omega.ffcv.train_reader(
            path=self.args.val_dataset_path,
            pipelines={"image": pipes[0], "label": pipes[1]},
            batch_size=self.args.batch_size,
            world_size=self.args.world_size,
        )
        # self.train_samples = train_loader.reader.num_samples
        # transforms = omega.transforms.imagenet_val_dataset()
        # data_source, self.num_classes = omega.dataset.get_dataset(
        #     self.args.dataset_path, transforms, split="test"
        # )
        # per_device_batch_size = self.args.batch_size // self.args.world_size
        # if self.args.world_size > 1:
        #     val_sampler = torch.utils.data.distributed.DistributedSampler(
        #         data_source, shuffle=False, drop_last=False
        #     )
        #     shuffle = None
        # else:
        #     val_sampler = None
        #     shuffle = False
        # val_loader = torch.utils.data.DataLoader(
        #     data_source,
        #     batch_size=per_device_batch_size,
        #     num_workers=10,
        #     pin_memory=True,
        #     sampler=val_sampler,
        #     shuffle=shuffle,
        # )
        return val_loader

    def initialize_modules(self):
        import torchmetrics
        import torchvision
        from torchmetrics.classification import MulticlassAccuracy

        self.valid_accuracy = MulticlassAccuracy(self.num_classes, top_k=1).to(
            self.this_device
        )
        self.valid_accuracy5 = MulticlassAccuracy(self.num_classes, top_k=5).to(
            self.this_device
        )
        self.train_loss = torchmetrics.MeanMetric().to(self.this_device)

        model = torchvision.models.__dict__[self.args.architecture]()
        model = omega.utils.replace_module(
            model, torch.nn.ReLU, MyReLU, None, p=self.args.proba
        )
        if self.args.projector_depth == 0:
            self.projector = torch.nn.Identity()
            if self.train_samples > 200000 and model.fc.in_features > 512:
                self.extra_classifier = torch.nn.Sequential(
                    torch.nn.Linear(
                        model.fc.in_features, 512, bias=False, dtype=torch.float32
                    ),
                    torch.nn.Linear(512, self.train_samples, dtype=torch.float32),
                )
            elif self.train_samples > 1100000 and model.fc.in_features > 256:
                self.extra_classifier = torch.nn.Sequential(
                    torch.nn.Linear(
                        model.fc.in_features, 256, bias=False, dtype=torch.float32
                    ),
                    torch.nn.Linear(256, self.train_samples, dtype=torch.float32),
                )
            else:
                self.extra_classifier = torch.nn.Linear(
                    model.fc.in_features, self.train_samples, dtype=torch.float32
                )
        else:
            w = self.args.projector_width
            projectors = []
            for l in range(self.args.projector_depth):
                fan_in = model.fc.in_features if l == 0 else w
                projectors.append(torch.nn.Linear(fan_in, w, bias=False))
                projectors.append(torch.nn.BatchNorm1d(w))
                projectors.append(torch.nn.ReLU())
            self.projector = torch.nn.Sequential(*projectors)
            self.extra_classifier = torch.nn.Linear(w, self.train_samples)
        self.classifier = torch.nn.Linear(model.fc.in_features, self.num_classes)
        model.fc = torch.nn.Identity()
        self.model = model

    def initialize_optimizer(self):
        lr = self.args.learning_rate
        wd = self.args.weight_decay
        p = self.parameters()
        if self.args.optimizer == "adamw":
            return optim.AdamW(p, weight_decay=wd, lr=lr, eps=1e-8, betas=(0.9, 0.999))
        elif self.args.optimizer == "adam":
            return optim.Adam(p, weight_decay=wd, lr=lr, eps=1e-8, betas=(0.9, 0.999))
        else:
            return optim.SGD(p, weight_decay=wd, lr=lr, momentum=self.args.momentum)

    def compute_loss(self):
        x = self.data[0]
        y = self.data[1][:, 0]
        z = self.data[1][:, 1]
        print(self.this_device, x.shape)
        # x = self.data[0].cuda(self.this_device, non_blocking=True)
        # y = self.data[1].cuda(self.this_device, non_blocking=True)
        # z = self.data[2].cuda(self.this_device, non_blocking=True)
        preds = self.model(x)
        preds_true = self.classifier(preds.detach())
        preds_false = self.extra_classifier(self.projector(preds))
        true_loss = torch.nn.functional.cross_entropy(preds_true, y)
        zonehot = torch.nn.functional.one_hot(z, self.train_samples).type_as(x)
        if self.args.loss == "ce":
            other_loss = torch.nn.functional.cross_entropy(
                preds_false, z, label_smoothing=self.args.label_smoothing
            )
        elif self.args.loss == "sce":
            reversed = torch.softmax(preds_false, dim=-1) * torch.log(
                zonehot.clip(0.001)
            )
            other_loss = (
                torch.nn.functional.cross_entropy(
                    preds_false, z, label_smoothing=self.args.label_smoothing
                )
                + reversed.sum(-1).mean()
            )
        elif self.args.loss == "l2":
            other_loss = torch.nn.functional.mse_loss(preds_false, zonehot)
        elif self.args.loss == "bce":
            other_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                preds_false, zonehot
            )
        elif self.args.loss == "l1":
            other_loss = torch.nn.functional.l1_loss(preds_false, zonehot)
        elif self.args.loss == "sboot":
            other_loss = other_loss = torch.nn.functional.cross_entropy(
                preds_false,
                zonehot * self.args.beta
                + torch.softmax(preds_false, dim=-1) * (1 - self.args.beta),
                label_smoothing=self.args.label_smoothing,
            )
        self.train_loss.update(other_loss)
        return other_loss + true_loss

    def eval_step(self):
        x = self.data[0]  # .cuda(self.this_device, non_blocking=True)
        y = self.data[1]  # .cuda(self.this_device, non_blocking=True)
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
        if self.args.epochs > 100:
            return
        if "tiny" in self.args.train_dataset_path.absolute().as_posix():
            if self.rank == 0 and self.epoch == 19 and accu < 0.08:
                afd
            elif self.rank == 0 and self.epoch == 99:
                if self.args.strength == 0 and accu < 0.16:
                    afd
                elif self.args.strength == 3 and accu < 0.27:
                    afd
        elif "Aircraft" in self.args.train_dataset_path.absolute().as_posix():
            if self.rank == 0 and self.epoch == 19 and accu < 0.03:
                afd
            elif self.rank == 0 and self.epoch == 99:
                if self.args.strength == 0 and accu < 0.07:
                    afd
                elif self.args.strength == 3 and accu < 0.12:
                    afd
        elif "Food101" in self.args.train_dataset_path.absolute().as_posix():
            if self.rank == 0 and self.epoch == 19:
                if self.args.strength == 0 and accu < 0.11:
                    afd
                elif self.args.strength == 3 and accu < 0.15:
                    afd
            elif self.rank == 0 and self.epoch == 99:
                if self.args.strength == 0 and accu < 0.17:
                    afd
                elif self.args.strength == 3 and accu < 0.27:
                    afd
        elif "CIFAR100" in self.args.train_dataset_path.absolute().as_posix():
            if self.rank == 0 and self.epoch == 19:
                if self.args.strength == 0 and accu < 0.15:
                    afd
                elif self.args.strength == 3 and accu < 0.2:
                    afd
            elif self.rank == 0 and self.epoch == 99:
                if self.args.strength == 0 and accu < 0.22:
                    afd
                elif self.args.strength == 3 and accu < 0.31:
                    afd
        elif "CIFAR10" in self.args.train_dataset_path.absolute().as_posix():
            if self.rank == 0 and self.epoch == 19:
                if self.args.strength == 0 and accu < 0.35:
                    afd
                elif self.args.strength == 3 and accu < 0.45:
                    afd
            elif self.rank == 0 and self.epoch == 99:
                if self.args.strength == 0 and accu < 0.50:
                    afd
                elif self.args.strength == 3 and accu < 0.65:
                    afd
        elif "OxfordIIITPet" in self.args.train_dataset_path.absolute().as_posix():
            if self.rank == 0 and self.epoch == 19:
                if self.args.strength == 0 and accu < 0.045:
                    afd
                elif self.args.strength == 3 and accu < 0.065:
                    afd
            elif self.rank == 0 and self.epoch == 99:
                if self.args.strength == 0 and accu < 0.14:
                    afd
                elif self.args.strength == 3 and accu < 0.23:
                    afd

    def after_train_step(self):
        self.scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--strength", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--proba", type=float, default=0.5)
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "l2", "l1", "sce", "bce", "sboot"],
    )
    parser.add_argument("--projector-depth", type=int, default=2)
    parser.add_argument("--projector-width", type=int, default=128)
    parser.add_argument(
        "--train-dataset-path",
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
            Path("/private/home/rbalestriero/DATASETS/CIFAR10/train_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv"),
            Path("/datasets01/places205/121517/pytorch/"),
            Path("/private/home/rbalestriero/DATASETS/IMAGENET/train_500_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"),
        ],
    )
    parser.add_argument(
        "--val-dataset-path",
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
            Path("/private/home/rbalestriero/DATASETS/CIFAR10/val_raw.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/CIFAR100/val_raw.ffcv"),
            Path("/datasets01/places205/121517/pytorch/"),
            Path("/private/home/rbalestriero/DATASETS/IMAGENET/val_500_jpg.ffcv"),
            Path("/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"),
        ],
    )
    omega.argparse.make_config(parser)

    args = parser.parse_args()
    args.grad_max_norm = 5.0
    args.eval_each_epoch = True

    # if args.load_config_from.is_file():
    #     import json
    #     with open(args.load_config_from, 'rt') as f:
    #         t_args = argparse.Namespace()
    #         t_args.__dict__.update(json.load(f))
    #         args = parser.parse_args(namespace=t_args)

    model = Model(args)
    executor = submitit.AutoExecutor(folder=model.args.folder, slurm_max_num_timeout=30)

    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_signal_delay_s=120,
        nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        tasks_per_node=args.gpus_per_node,
        cpus_per_task=10,
        slurm_partition="learnlab",
        name=args.process_name,
        slurm_setup=[
            "module purge",
            "module load anaconda3/2020.11-nold",
            "source activate /private/home/rbalestriero/.conda/envs/ffcv",
        ],
    )
    if args.train_dataset_path == "/datasets01/imagenet_full_size/061417/":
        executor.update_parameters(slurm_constraint="volta32gb")
    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
