"""
File: write_dataset.py
Project: omega
-----
# Copyright (c) Randall Balestriero.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
import torch
import omega
from glob import glob
import submitit
from pathlib import Path

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


class Model(omega.Trainer):
    def initialize_train_loader(self):
        name = self.args.train_dataset_path.parent.name.lower()
        self.num_classes, self.image_size = omega.dataset.NAME_TO_CLASS[name]
        pipes = omega.transforms.ffcv_imagenet_train_dataset(
            device=self.this_device,
            dtype=torch.float16 if self.args.float16 else torch.float32,
            strength=self.args.strength,
            size=self.image_size,
            ratio=224 / 256,
        )

        train_loader = omega.ffcv.train_reader(
            path=self.args.train_dataset_path,
            pipelines={"image": pipes[0], "label": pipes[1]},
            batch_size=self.args.batch_size,
            world_size=self.args.world_size,
        )
        self.train_samples = train_loader.reader.num_samples
        return train_loader

    def initialize_val_loader(self):
        pipes = omega.transforms.ffcv_imagenet_val_dataset(
            device=self.this_device,
            dtype=torch.float16 if self.args.float16 else torch.float32,
            size=self.image_size,
            ratio=224 / 256,
        )

        val_loader = omega.ffcv.train_reader(
            path=self.args.val_dataset_path,
            pipelines={"image": pipes[0], "label": pipes[1]},
            batch_size=self.args.batch_size,
            world_size=self.args.world_size,
        )
        return val_loader

    def initialize_modules(self):
        from torchmetrics.classification import MulticlassAccuracy

        self.valid_accuracy = MulticlassAccuracy(self.num_classes, top_k=1).to(
            self.this_device
        )
        self.valid_accuracy5 = MulticlassAccuracy(self.num_classes, top_k=5).to(
            self.this_device
        )
        paths = glob(self.args.path_to_models)[: self.args.max_num_models]
        print("Found ", len(paths), " paths")
        print("Only using", self.args.max_num_models)
        for i, path in enumerate(paths):
            self._initialize_modules(f"model_{i}", path)
        self.num_modules = len(paths)
        self.classifier = torch.nn.Linear(self.fan_in * len(paths), self.num_classes)

    def _initialize_modules(self, name, path):
        model, self.fan_in = omega.utils.load_without_classifier(self.args.architecture)
        ckpt = torch.load(path, map_location="cpu")["model"]
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        model.requires_grad_(False)
        setattr(self, name, model)

    def compute_loss(self):
        x = self.data[0]
        y = self.data[1][:, 0]
        with torch.no_grad():
            outs = []
            for i in range(self.num_modules):
                outs.append(getattr(self, f"model_{i}")(x))
            outs = torch.cat(outs, 1)
        preds = self.classifier(outs)
        return torch.nn.functional.cross_entropy(
            preds, y, label_smoothing=self.args.label_smoothing
        )

    def eval_step(self):
        x = self.data[0]
        y = self.data[1]
        outs = []
        for i in range(self.num_modules):
            outs.append(getattr(self, f"model_{i}")(x))
        outs = torch.cat(outs, 1)
        preds = self.classifier(outs)
        self.valid_accuracy.update(preds, y)
        self.valid_accuracy5.update(preds, y)

    def after_eval_epoch(self):
        super().after_eval_epoch()
        self.log_txt(
            "eval_accuracies",
            accus=self.valid_accuracy.compute().item(),
            accus5=self.valid_accuracy5.compute().item(),
        )
        # Reset metric states after each epoch
        self.valid_accuracy.reset()
        self.valid_accuracy5.reset()

    def before_train_epoch(self):
        super().before_train_epoch()
        self.eval()


if __name__ == "__main__":

    parser = ArgumentParser(description="DDP Training")
    parser.add_argument("--strength", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--path-to-models", type=str, required=True)
    parser.add_argument("--max-num-models", type=int, default=32)
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
    print(args.path_to_models)
    print(glob(args.path_to_models)[: args.max_num_models])

    model = Model(args)
    executor = submitit.AutoExecutor(
        folder=model.args.folder, slurm_max_num_timeout=450
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
    # executor.update_parameters(slurm_constraint="volta32gb")
    job = executor.submit(model)
    print(f"Submitted job_id: {job.job_id}")
