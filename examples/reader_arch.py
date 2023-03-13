from omega import reader
from matplotlib import pyplot as plt
import numpy as np


# get colormap
cmap = plt.cm.tab20
# build cycler with 5 equally spaced colors from that colormap
colors = {
    "5e-05/0.0": cmap(0),
    "5e-05/0.001": cmap(1),
    "5e-05/0.01": cmap(2),
    "0.0002/0.0": cmap(3),
    "0.0002/0.001": cmap(4),
    "0.0002/0.01": cmap(5),
    "0.001/0.0": cmap(6),
    "0.001/0.001": cmap(7),
    "0.001/0.01": cmap(8),
    "0.001/0.05": cmap(9),
    "0.001/0.2": cmap(10),
    "5e-05/0.05": "k",
    "0.0002/0.05": "gray",
    "1e-05/0.05": "red",
}
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
        "legend.fontsize": 12,
        "legend.handlelength": 0.8,
        "legend.handleheight": 1,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.1,
        "legend.markerscale": 4,
        "legend.labelcolor": "linecolor",
    }
)


def plot_CIFAR(dataset="CIFAR100"):
    col = {
        "resnet18": 0,
        "resnet34": 1,
        "resnet50": 2,
        "resnet101": 3,
        "resnet152": 4,
        "alexnet": 5,
    }
    perfs = {
        "resnet18": [],
        "resnet34": [],
        "resnet50": [],
        "resnet101": [],
        "resnet152": [],
        "alexnet": [],
    }
    scatters_X = []
    scatters_y = []
    colors = []
    cmap = plt.cm.plasma
    fig1, axs1 = plt.subplots(
        2, len(col), sharey="row", sharex="all", figsize=(4 * len(col), 4 * 2)
    )

    runs = reader.gather_runs(f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_{dataset}_3/")

    for run in runs:
        if run["hparams"]["strength"] != "3":
            continue
        c = col[run["hparams"]["architecture"]]
        label = f'{run["hparams"]["learning_rate"]}/{run["hparams"]["weight_decay"]}'
        perfs[run["hparams"]["architecture"]].append(
            run["json_log"]["accus"].max() * 100
        )
        scatters_X.extend(list(run["json_log"]["accus"] * 100))
        scatters_y.extend(list(run["json_log"]["train_loss"]))
        colors.extend(
            [cmap(float(run["hparams"]["label_smoothing"]))]
            * len(run["json_log"]["accus"])
        )
        # print(
        #     run["hparams"]["architecture"],
        #     len(run["json_log"]["accus"]),
        #     run["hparams"]["learning_rate"],
        #     run["hparams"]["weight_decay"],
        #     run["json_log"]["accus"].max() * 100,
        # )
        # axs1[0, c].plot(
        #     run["json_log"]["accus"].ewm(span=11).mean() * 100, c=colors[label]
        # )
        # axs1[0, c].set_title(run["hparams"]["architecture"], style="italic")
        # axs1[1, c].plot(
        #     run["json_log"]["accus5"].ewm(span=11).mean() * 100,
        #     c=colors[label],
        #     label=label if c == 2 else None,
        # )
        # axs1[1, c].set_xlabel("epochs")

    legend = axs1[1, 2].legend(ncol=2, title="lr/wd")
    plt.setp(legend.get_title(), color="0.5")
    axs1[0, 0].set_ylim([20, 65])
    axs1[1, 0].set_ylim([40, 90])
    axs1[0, 0].set_xlim([0, 4000])
    axs1[0, 0].set_xticks([1000, 2000, 3000, 4000])
    axs1[0, 0].set_ylabel("top1 test acc. (%)")
    axs1[1, 0].set_ylabel("top5 test acc. (%)")
    fig1.savefig(f"saved_arch_{dataset}.png")
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(
        scatters_X,
        scatters_y,
        c=colors,
        s=5,
        # edgecolors="black",
        # linewidths=0.3,
        alpha=0.1,
    )
    # axs.set_yscale("log")
    axs.set_ylabel("DIET train loss")
    axs.set_xlabel("top1 test acc. (%)")
    axs.set_title(dataset, style="italic")
    fig.savefig(f"loss_accu_{dataset}.png", dpi=140)
    plt.close()

    # for key, value in perfs.items():
    #     print(key, max(value))


def plot_IMAGENET(dataset="TINYIMAGENET"):
    col = {
        "resnet18": (0, 0),
        "resnet34": (0, 1),
        "resnet50": (0, 2),
        "wide_resnet50_2": (0, 3),
        "resnext50_32x4d": (1, 0),
        "densenet121": (1, 1),
        "convnext_tiny": (1, 2),
        "convnext_small": (1, 3),
        "resnet101": (2, 0),
        "MLPMixer": (2, 1),
        "swin_t": (2, 2),
        "swin_s": (2, 3),
        "vit_b_16": (1, 4),
    }
    perfs = {
        "resnet18": [],
        "resnet34": [],
        "resnet50": [],
        "wide_resnet50_2": [],
        "resnext50_32x4d": [],
        "regnet_x_16gf": [],
        "densenet121": [],
        "convnext_tiny": [],
        "convnext_small": [],
        "resnet101": [],
        "MLPMixer": [],
        "swin_t": [],
        "swin_s": [],
        "vit_b_16": [],
    }
    names = []
    scatters_X = []
    scatters_y = []
    colors = []

    fig, axs = plt.subplots(3, 5, sharey="all", sharex="all", figsize=(6 * 4, 3 * 4))
    runs = reader.gather_runs(f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_{dataset}_3")

    cmap = plt.cm.plasma
    for run in runs:

        print(run["hparams"]["epochs"])
        if run["hparams"]["architecture"] not in perfs:
            print(run["hparams"]["architecture"])
            continue
        perfs[run["hparams"]["architecture"]].append(
            run["json_log"]["accus"].max() * 100
        )
        if run["hparams"]["architecture"] == "resnet50":
            names.append(run["path"])
        c = col[run["hparams"]["architecture"]]
        label = f'{run["hparams"]["learning_rate"]}/{run["hparams"]["weight_decay"]}'
        axs[c[0], c[1]].plot(
            100 * run["json_log"]["epoch"] / int(run["hparams"]["epochs"]),
            run["json_log"]["accus"].ewm(span=7).mean() * 100,
            c="k",
        )
        axs[c[0], c[1]].set_title(run["hparams"]["architecture"], style="italic")
        scatters_X.extend(list(run["json_log"]["accus"] * 100))
        scatters_y.extend(list(run["json_log"]["train_loss"]))
        colors.extend(
            [cmap(float(run["hparams"]["label_smoothing"]))]
            * len(run["json_log"]["accus"])
        )
        # axs[c[0], c[1]].set_ylim(15, run["json_log"]["accus"].max() * 100)

    for c in range(3):
        axs[c, 0].set_ylabel("top1 test acc. (%)")
    for c in range(4):
        axs[-1, c].set_xlabel("epochs (%)")
    # axs[0, 0].set_ylim([30, 75])
    axs[0, 0].set_xlim([0, 100])

    fig.savefig(f"saved_arch_{dataset}.png")
    plt.close()
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(
        scatters_X,
        scatters_y,
        c=colors,
        s=5,
        # edgecolors="black",
        # linewidths=0.3,
        alpha=0.1,
    )
    # axs.set_yscale("log")
    axs.set_xlabel("top1 test acc. (%)")
    axs.set_title(dataset, style="italic")
    fig.savefig(f"loss_accu_{dataset}.png", dpi=140)
    plt.close()

    for key, value in perfs.items():
        if len(value) == 0:
            continue
        print(
            "\em\cellcolor{blue!45}",
            key.replace("_", "\_"),
            "& \cellcolor{red!15}",
            np.round(max(value), 2),
            r"\\",
        )
    print("BEST:", names[np.argmax(perfs["resnet50"])])


if __name__ == "__main__":
    plot_CIFAR()
    plot_IMAGENET("TINYIMAGENET")
    plot_IMAGENET("IMAGENET100")
