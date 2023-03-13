import torch
import argparse
import submitit
import omega
import numpy as np
from pathlib import Path

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


class Model(omega.Trainer):
    def initialize_train_loader(self):
        name = self.args.train_dataset_path.parent.name.lower()
        self.num_classes, self.image_size = omega.dataset.NAME_TO_CLASS[name]
        if self.args.architecture in ["swin_t", "vit_b_16"]:
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
        index_to_class = torch.range(0, len(train_loader.indices) - 1)
        if len(train_loader.indices) > self.args.max_indices:
            indices = np.random.RandomState(0).permutation(len(train_loader.indices))[
                : self.args.max_indices
            ]
            for i, j in enumerate(indices):
                index_to_class[j] = i
            train_loader = omega.ffcv.train_reader(
                path=self.args.train_dataset_path,
                pipelines={"image": pipes[0], "label": pipes[1]},
                batch_size=self.args.batch_size,
                world_size=self.args.world_size,
                indices=indices,
            )
        self.train_samples = len(train_loader.indices)
        if self.args.indices_from.is_file():
            index_to_class = np.load(self.args.indices_from, allow_pickle=True)[
                "indices"
            ]
            self.train_samples = int(index_to_class.max() + 1)
            index_to_class = torch.from_numpy(index_to_class)
        self.register_buffer(
            "index_to_class", index_to_class.to(self.this_device).long()
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
        import torchvision
        from torchmetrics.classification import MulticlassAccuracy

        self.valid_accuracy = MulticlassAccuracy(self.num_classes, top_k=1).to(
            self.this_device
        )
        self.valid_accuracy5 = MulticlassAccuracy(self.num_classes, top_k=5).to(
            self.this_device
        )
        self.train_loss = torchmetrics.MeanMetric().to(self.this_device)

        is_cifar = "CIFAR" in self.args.train_dataset_path.absolute().as_posix()
        if self.args.architecture == "MLPMixer":
            from mlp_mixer_pytorch import MLPMixer

            model = MLPMixer(
                image_size=self.image_size,
                channels=3,
                patch_size=max(4, self.image_size // 16),
                dim=512,
                depth=8 if is_cifar else 12,
                num_classes=1000,
            )
        else:
            model = torchvision.models.__dict__[self.args.architecture]()
            if is_cifar and "resnet" in self.args.architecture:
                model.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                model.maxpool = torch.nn.Identity()
        if "MLPMixer" in self.args.architecture:
            fan_in = model[16].in_features
            model[16] = torch.nn.Identity()
        elif "alexnet" in self.args.architecture:
            fan_in = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Identity()
        elif "convnext" in self.args.architecture:
            fan_in = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Identity()
        elif "convnext" in self.args.architecture:
            fan_in = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Identity()
        elif (
            "resnet" in self.args.architecture
            or "resnext" in self.args.architecture
            or "regnet" in self.args.architecture
        ):
            fan_in = model.fc.in_features
            model.fc = torch.nn.Identity()
        elif "densenet" in self.args.architecture:
            fan_in = model.classifier.in_features
            model.classifier = torch.nn.Identity()
        elif "mobile" in self.args.architecture:
            fan_in = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Identity()
        elif "vit" in self.args.architecture:
            fan_in = model.heads.head.in_features
            model.heads.head = torch.nn.Identity()
        elif "swin" in self.args.architecture:
            fan_in = model.head.in_features
            model.head = torch.nn.Identity()

        self.extra_classifier = torch.nn.Linear(fan_in, self.train_samples)
        print(self.train_samples)
        self.classifier = torch.nn.Linear(fan_in, self.num_classes)
        self.model = model

    def compute_loss(self):

        c = lambda a, b: torch.nn.functional.cross_entropy(
            a, b, label_smoothing=self.args.label_smoothing
        )
        c2 = lambda a, b: torch.nn.functional.cross_entropy(a, b, label_smoothing=0.1)
        x = self.data[0]
        labels = self.data[1][:, 0]
        data_labels = self.index_to_class[self.data[1][:, 1]]
        applied = False
        if self.args.aggressive:
            if np.random.rand() < 0.5:
                applied = True
                if np.random.rand() < 0.5:
                    x, yz_a, yz_b, lam = self.cutmix_data(x, self.data[1])
                else:
                    x, yz_a, yz_b, lam = self.mixup_data(x, self.data[1])

        preds = self.model(x)
        preds_true = self.classifier(preds.detach())
        preds_false = self.extra_classifier(preds)

        if not applied:
            true_loss = c2(preds_true, labels)
            other_loss = c(preds_false, data_labels)
        else:
            true_loss = self.cutmix_criterion(
                c2, preds_true, yz_a[:, 0], yz_b[:, 0], lam
            )
            other_loss = self.cutmix_criterion(
                c,
                preds_false,
                self.index_to_class[yz_a[:, 1]],
                self.index_to_class[yz_b[:, 1]],
                lam,
            )

        self.train_loss.update(other_loss)
        return other_loss + true_loss

    def eval_step(self):
        x = self.data[0]
        y = self.data[1]
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

    def after_train_step(self):
        self.scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--strength", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument("--max-indices", type=int, default=999999999)
    parser.add_argument("--indices-from", type=Path, default="./")
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
    args.grad_max_norm = 5.0
    args.eval_each_epoch = True

    # if args.load_config_from.is_file():
    #     import json
    #     with open(args.load_config_from, 'rt') as f:
    #         t_args = argparse.Namespace()
    #         t_args.__dict__.update(json.load(f))
    #         args = parser.parse_args(namespace=t_args)

    model = Model(args)
    model()
    asdf
    executor = submitit.AutoExecutor(
        folder=model.args.folder, slurm_max_num_timeout=150
    )

    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_signal_delay_s=250,
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
    # if args.train_dataset_path == "/datasets01/imagenet_full_size/061417/":
    executor.update_parameters(slurm_constraint="volta32gb")
    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
