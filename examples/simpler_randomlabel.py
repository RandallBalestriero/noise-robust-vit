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
        N = len(train_loader.indices)
        index_to_class = torch.arange(N)
        if N > self.args.max_indices and self.args.entropy_propagation == 0.0:
            rng = np.random.RandomState(self.args.indices_seed)
            indices = rng.permutation(N)[: self.args.max_indices]
            for i, j in enumerate(indices):
                index_to_class[j] = i
            train_loader = omega.ffcv.train_reader(
                path=self.args.train_dataset_path,
                pipelines={"image": pipes[0], "label": pipes[1]},
                batch_size=self.args.batch_size,
                world_size=self.args.world_size,
                indices=indices,
            )
            N = len(train_loader.indices)
        elif N > self.args.max_indices and self.args.entropy_propagation > 0:
            rng = np.random.RandomState(self.args.indices_seed)
            indices = rng.permutation(N)[: self.args.max_indices]
            index_to_class.copy_(-1)
            for i, j in enumerate(indices):
                index_to_class[j] = i
        self.train_samples = N
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
            fan_in = model[16].in_features
            model[16] = torch.nn.Identity()
        else:
            model, fan_in = omega.utils.load_without_classifier(self.args.architecture)

        if is_cifar and "resnet" in self.args.architecture:
            model.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            model.maxpool = torch.nn.Identity()

        if not self.args.supervised:
            if self.args.projector_depth == 0:
                self.extra_classifier = torch.nn.Linear(
                    fan_in,
                    self.args.clip_output_dim if self.args.clip else self.train_samples,
                )
            else:
                w = self.args.projector_width
                layers = [torch.nn.Linear(fan_in, w)]
                for _ in range(self.args.projector_depth):
                    layers.append(torch.nn.BatchNorm1d(w))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Linear(w, w, bias=False))
                layers.pop(-1)
                layers.append(
                    torch.nn.Linear(
                        w,
                        self.args.clip_output_dim
                        if self.args.clip
                        else self.train_samples,
                    )
                )
                self.extra_classifier = torch.nn.Sequential(*layers)
        if self.args.clip:
            layers = [torch.nn.Linear(22, 512)]
            layers.append(torch.nn.BatchNorm1d(512))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(512, 512, bias=False))
            layers.append(torch.nn.BatchNorm1d(512))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(512, self.args.clip_output_dim))
            self.index_encoding = torch.nn.Sequential(*layers)
            # self.temperature = torch.nn.Linear(1,1,bias=False)
            # torch.nn.init.constant_(self.temperature.weight,0.1)
        self.classifier = torch.nn.Linear(fan_in, self.num_classes)
        self.model = model

    def compute_loss(self):
        x = self.data[0]
        labels, indices = self.data[1].unbind(1)
        indices = self.index_to_class[indices]
        preds = self.model(x)
        if self.args.supervised:
            loss = torch.nn.functional.cross_entropy(self.classifier(preds), labels)
            self.train_loss.update(loss)
            return loss

        # online probe
        preds_true = self.classifier(preds.detach())
        true_loss = torch.nn.functional.cross_entropy(preds_true, labels)

        # DIET
        if self.args.clip:
            mask = 2 ** torch.arange(22).to(self.this_device, torch.int)
            bins = (
                indices.int()
                .unsqueeze(-1)
                .bitwise_and(mask)
                .ne(0)
                .byte()
                .type(preds.dtype)
                - 0.5
            )
            index_preds = self.index_encoding(bins)
            sim = (
                torch.nn.functional.cosine_similarity(
                    index_preds[None], self.extra_classifier(preds)[:, None], dim=2
                )
                / self.args.temperature
            )
            labels = torch.arange(index_preds.size(0), device=self.this_device)
            loss_t = torch.nn.functional.cross_entropy(
                sim, labels, label_smoothing=self.args.label_smoothing
            )
            loss_i = torch.nn.functional.cross_entropy(
                sim.T, labels, label_smoothing=self.args.label_smoothing
            )
            other_loss = (loss_t + loss_i) / 2.0
        else:
            preds_false = self.extra_classifier(preds)
            other_loss = torch.nn.functional.cross_entropy(
                preds_false, indices, label_smoothing=self.args.label_smoothing
            )

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

    def initialize_optimizer(self):
        if self.args.lr_scaling == 1.0 and self.args.wd_scaling == 1.0:
            return super().initialize_optimizer()
        params = list(self.parameters())
        if not self.args.clip:
            W = self.extra_classifier.weight
            params.remove(W)
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": params,
                        "lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [W],
                        "lr": self.args.learning_rate * self.args.lr_scaling,
                        "weight_decay": self.args.weight_decay * self.args.wd_scaling,
                    },
                ],
                eps=1e-8,
                betas=(self.args.beta1, self.args.beta2),
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                eps=1e-8,
                betas=(self.args.beta1, self.args.beta2),
            )
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--strength", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-indices", type=int, default=999999999)
    parser.add_argument("--indices-seed", type=int, default=0)
    parser.add_argument("--indices-from", type=Path, default="./")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--lr-scaling", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip-output-dim", type=int, default=256)
    parser.add_argument("--wd-scaling", type=float, default=1.0)
    parser.add_argument("--projector-depth", type=int, default=0)
    parser.add_argument("--projector-width", type=int, default=1024)
    parser.add_argument("--entropy-propagation", type=float, default=0.0)
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
    # args.grad_max_norm = 55.0
    # args.eval_each_epoch = True

    # if args.load_config_from.is_file():
    #     import json
    #     with open(args.load_config_from, 'rt') as f:
    #         t_args = argparse.Namespace()
    #         t_args.__dict__.update(json.load(f))
    #         args = parser.parse_args(namespace=t_args)

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
    # if args.train_dataset_path == "/datasets01/imagenet_full_size/061417/":
    # executor.update_parameters(slurm_constraint="volta32gb")
    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
