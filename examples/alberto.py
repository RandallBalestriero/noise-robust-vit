import torch
import argparse
import omega
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchmetrics import Metric, CatMetric
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


class MyAggregator(Metric):
    def __init__(self, ndim):
        super().__init__()
        self.add_state("correct", default=torch.zeros(0, ndim))

    def update(self, preds: torch.Tensor):
        self.correct = torch.cat([self.correct, preds])

    def compute(self):
        return self.correct


def VICReg(preds):
    N, D = preds.shape
    I = torch.eye(
        D,
        device=preds.device,
        dtype=preds.dtype,
    )
    m = preds.mean(0)
    cov = (preds - m).T @ (preds - m) / N
    VCreg = cov.sub(I).square().mean()
    Ireg = (preds[: N // 2] - preds[N // 2 :]).square().mean()
    return VCreg, Ireg


class PositivePairs(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        x, y = self.dataset[i]
        x2, y = self.dataset[i]
        return x, x2, y

    def __len__(self):
        return len(self.dataset)


class Model(omega.Trainer):
    def initialize_train_loader(self):
        if self.args.augmentation == "crop":
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomAffine(0, [0.1, 0.1]),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif self.args.augmentation == "none":
            t = torchvision.transforms.ToTensor()
        elif self.args.augmentation == "rotation":
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomRotation(90),
                    torchvision.transforms.ToTensor(),
                ]
            )
        dataset = torchvision.datasets.MNIST(
            "/private/home/rbalestriero/DATASETS/MNIST/",
            train=True,
            transform=t,
        )
        dataset = PositivePairs(dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        return train_loader

    def initialize_val_loader(self):
        dataset = torchvision.datasets.MNIST(
            "/private/home/rbalestriero/DATASETS/MNIST/",
            train=False,
            transform=torchvision.transforms.ToTensor(),
        )
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, num_workers=10
        )
        return val_loader

    def initialize_modules(self):
        import torchmetrics
        from torchmetrics.classification import MulticlassAccuracy

        self.valid_accuracy = MulticlassAccuracy(10, top_k=1).to(self.this_device)
        self.valid_accuracy5 = MulticlassAccuracy(10, top_k=5).to(self.this_device)
        self.train_loss = torchmetrics.MeanMetric().to(self.this_device)
        self.vicreg_loss = torchmetrics.MeanMetric().to(self.this_device)
        self.vicreg_test = torchmetrics.MeanMetric().to(self.this_device)
        self.aggregator_x = MyAggregator(self.args.embedding_dim)
        self.aggregator_y = CatMetric()

        if self.args.model == "CNN":
            self.model = torch.nn.DataParallel(
                torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, 3),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(32, 128, 3, stride=2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(128, 512, 5, stride=2),
                    torch.nn.LeakyReLU(),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(512, self.args.embedding_dim),
                )
            )
        elif self.args.model == "MLP":
            self.model = torch.nn.DataParallel(
                torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(28 * 28, 512),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(512, self.args.embedding_dim),
                )
            )
        else:
            self.model = torch.nn.DataParallel(
                torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(28 * 28, self.args.embedding_dim),
                )
            )
        self.classifier = torch.nn.Linear(self.args.embedding_dim, 10)

    def compute_loss(self):
        X = self.data[0].to(self.this_device, non_blocking=True)
        X2 = self.data[1].to(self.this_device, non_blocking=True)
        y = self.data[2].to(self.this_device, non_blocking=True)
        inputs = torch.cat([X, X2])
        if self.step == 0 and self.epoch == 0:
            grid1 = torchvision.utils.make_grid(X[:64].cpu(), nrow=8).numpy()
            grid2 = torchvision.utils.make_grid(X2[:64].cpu(), nrow=8).numpy()
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(grid1.transpose(1, 2, 0), aspect="auto")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.imshow(grid2.transpose(1, 2, 0), aspect="auto")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(self.args.folder / "views.png")
            plt.close()

        preds = self.model(inputs)
        VCreg, Ireg = VICReg(preds)
        ssl_loss = VCreg + self.args.lamb * Ireg

        y = torch.cat([y, y])
        loss = torch.nn.functional.cross_entropy(self.classifier(preds.detach()), y)
        self.vicreg_loss.update(ssl_loss)
        self.train_loss.update(loss)
        self.aggregator_x.update(preds)
        self.aggregator_y.update(y)
        return loss + 1000 * ssl_loss

    def eval_step(self):
        x = self.data[0].to(self.this_device, non_blocking=True)
        y = self.data[1].to(self.this_device, non_blocking=True)
        preds = self.classifier(self.model(x))
        VCreg, Ireg = VICReg(preds)
        self.vicreg_test.update(VCreg + self.args.lamb * Ireg)
        self.valid_accuracy.update(preds, y)
        self.valid_accuracy5.update(preds, y)

    def after_eval_epoch(self):
        super().after_eval_epoch()
        self.log_txt(
            "eval_accuracies",
            accus=self.valid_accuracy.compute().item(),
            accus5=self.valid_accuracy5.compute().item(),
            train_loss=self.train_loss.compute().item(),
            VICReg_loss=self.vicreg_loss.compute().item(),
            VICReg_test=self.vicreg_test.compute().item(),
        )
        # Reset metric states after each epoch
        self.valid_accuracy.reset()
        self.valid_accuracy5.reset()
        self.vicreg_loss.reset()
        self.vicreg_test.reset()
        self.train_loss.reset()

    def after_train_step(self):
        self.scheduler.step()

    def before_train_epoch(self):
        super().before_train_epoch()
        if self.args.model == "CNN":
            w = self.model.module[0].weight.detach().cpu().numpy()
            np.savez(self.args.folder / "filters.npz", filters=w)

    def after_train_epoch(self):
        super().after_train_epoch()
        preds = self.aggregator_x.compute()
        y = self.aggregator_y.compute()
        self.aggregator_y.reset()
        self.aggregator_x.reset()
        np.savez(self.args.folder / "embeddings.npz", prds=preds.cpu(), y=y.cpu())


if __name__ == "__main__":

    from glob import glob
    from sklearn.manifold import TSNE
    import matplotlib

    tsne = TSNE(n_jobs=-1)
    paths = glob("ALBERTO/*/*/*/embeddings.npz")
    # for path in paths:
    #     print(path)
    #     data = np.load(path)
    #     if "tsne" in data:
    #         continue
    #     x = tsne.fit_transform(data["prds"][::16])
    #     np.savez(path, prds=data["prds"], y=data["y"], tsne=x)

    # cmap = matplotlib.cm.tab10
    # for path in paths:
    #     print(path)
    #     data = np.load(path)
    #     if "tsne" not in data:
    #         continue
    #     x = data["tsne"]
    #     y = data["y"][::16].astype("int")
    #     print(y)
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(
    #         x[:, 0],
    #         x[:, 1],
    #         c=[cmap(i) for i in y],
    #         alpha=0.5,
    #         linewidths=1,
    #         edgecolors="black",
    #     )
    #     details = path.split("/")[-4:-1]
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.tight_layout()
    #     plt.savefig(path.replace("embeddings.npz", "_".join(details) + "_tsne.png"))
    #     plt.close()
    # asdf
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--augmentation", type=str, default="crop")
    parser.add_argument("--model", type=str, default="CNN")
    omega.argparse.make_config(parser)

    args = parser.parse_args()
    args.eval_each_epoch = True
    args.weight_decay = 0

    model = Model(args)
    model()
