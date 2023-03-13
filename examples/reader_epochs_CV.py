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

epoch_subplot = {
    50: 0,
    100: 1,
    200: 2,
    500: 3,
    1000: 4,
    5000: 5,
    10000: 6,
}


def plot(arch="resnet18"):

    perfs = {
        50: [],
        100: [],
        200: [],
        500: [],
        1000: [],
        5000: [],
        10000: [],
    }
    fig, axs = plt.subplots(1, 7, figsize=(5 * 4, 7), sharey="all")

    for c, folder in zip([cmap(0.5), cmap(0.7), cmap(0.95)], ["", "LS_", "LSLS_"]):
        runs = reader.gather_runs(
            f"/checkpoint/rbalestriero/REVOLUTION2/EPOCHS_{folder}CV_CIFAR100/"
        )
        for run in runs:
            key = int(run["hparams"]["epochs"])
            if run["hparams"]["architecture"] != arch or key == 20000:
                continue
            perfs[key].append(run["json_log"]["accus"].max() * 100)
            if int(run["hparams"]["epochs"]) - 1 > run["json_log"]["epoch"].iloc[-1]:
                continue
            i = epoch_subplot[key]
            accu = run["json_log"]["accus"].ewm(span=1 + key // 700).mean()
            axs[i].plot(accu.to_numpy() * 100, c=c)
            axs[i].set_xlim(0, key)
            axs[i].set_ylim(20, 71)

        axs[0].set_ylabel("top1 test acc. (%)")
        for j in range(7):
            axs[j].set_xlabel("epoch")

    fig.savefig(f"saved_EPOCH_CV_{arch}.png", dpi=100)
    plt.close()
    return list(perfs.keys()), [np.round(np.max(i), 2) for i in perfs.values()]


if __name__ == "__main__":
    x, y1 = plot("resnet18")
    _, y2 = plot("resnet50")
    _, y3 = plot("resnet101")
    print(x, y1, y2, y3)
    X = np.stack([x, y1, y2, y3])
    X = np.concatenate(
        [np.array(["Epochs", "resnet18", "resnet50", "resnet101"])[:, None], X], 1
    )
    print(tabulate.tabulate(X, tablefmt="latex"))
