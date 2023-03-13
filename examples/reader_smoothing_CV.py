from omega import reader
from matplotlib import pyplot as plt
import numpy as np
import tabulate


# get colormap
cmap = plt.cm.Blues
plt.rcParams.update(
    {
        "font.size": 14,
        "figure.autolayout": True,
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.titlecolor": "0.5",
        "axes.labelsize": 20,
        "axes.labelcolor": "0.5",
        "axes.labelweight": "bold",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.grid": True,
        "grid.color": "0.5",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
    }
)


def plot(dataset, archs, axs):
    finals = []
    for arch, ax in zip(archs, axs):
        runs = reader.gather_runs(
            f"/checkpoint/rbalestriero/REVOLUTION2/LS_CV_{dataset}/"
        )
        x = np.array([])
        y = np.array([])
        z = []
        for run in runs:
            if run["hparams"]["architecture"] == arch:
                x = np.append(x, float(run["hparams"]["label_smoothing"]))
                y = np.append(y, run["json_log"]["accus"].max() * 100)
                z.append(run["json_log"]["accus"].ewm(span=11).mean())
        s = np.argsort(x)
        for i, j in enumerate(s):
            ax.plot(z[j] * 100, c=cmap((i + 1) / len(s)))
        finals.append(np.round(y[s], 2))
        ax.set_ylim(25, 70)
        ax.set_xlim(0, 4000)
        ax.set_title(arch, style="italic")
        ax.set_xlabel("epoch")
        if arch == archs[0]:
            ax.set_ylabel("top1 test acc. (%)")
    return x[s], finals


if __name__ == "__main__":
    archs = ["resnet18", "resnet50", "resnet101"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey="all", sharex="all")
    x, ys = plot("CIFAR100", archs, axs)
    X = np.stack([x] + ys)
    X = np.concatenate(
        [np.array(["Label smoothing"] + archs)[:, None], X],
        1,
    )
    print(tabulate.tabulate(X, tablefmt="latex"))
    fig.savefig("saved_smoothing_CV_CIFAR100.png")
    plt.close()

    archs = ["convnext_tiny", "convnext_small", "convnext_base"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey="all", sharex="all")
    x, ys = plot("TINYIMAGENET", archs, axs)
    X = np.stack([x] + ys)
    X = np.concatenate(
        [np.array(["Label smoothing"] + archs)[:, None], X],
        1,
    )
    print(tabulate.tabulate(X, tablefmt="latex"))
    fig.savefig("saved_smoothing_CV_TINYIMAGENET.png")
    plt.close()
