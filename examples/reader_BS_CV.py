from omega import reader
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate

# get colormap
cmap = plt.cm.tab10
plt.rcParams.update(
    {
        "font.size": 14,
        "figure.autolayout": True,
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.titlecolor": "0.5",
        "axes.labelsize": 18,
        "axes.labelcolor": "0.5",
        "axes.labelweight": "bold",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.grid": True,
        "grid.color": "0.5",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "legend.fontsize": 15,
        "legend.handlelength": 2.5,
        "legend.handleheight": 1,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 0.5,
        "legend.labelspacing": 0.1,
        # "legend.markerscale": 2,
        "legend.labelcolor": "linecolor",
    }
)


def plot(dataset="CIFAR100"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex="all", height_ratios=[2, 1])

    # runs = reader.gather_runs(
    #     f"/checkpoint/rbalestriero/REVOLUTION2/BS_CV_LS_{dataset}_0/"
    # )
    # x = np.array([])
    # y = np.array([])
    # y2 = np.array([])
    # z = np.array([])
    # for run in runs:
    #     print(run["json_log"]["epoch"].iloc[-1])
    #     x = np.append(x, int(run["hparams"]["batch_size"]))
    #     y = np.append(y, run["json_log"]["accus"].iloc[:31].max() * 100)
    #     y2 = np.append(y2, run["json_log"]["accus"].max() * 100)
    #     z = np.append(
    #         z, np.median(np.diff(run["json_log"]["relative_time"].iloc[2:].to_numpy()))
    #     )
    # s = np.argsort(x)
    # z = z[s]
    # axs[0].plot(
    #     x[s], y[s], "--o", c=cmap(0), label="lr:0.001 (epoch30)", alpha=0.7, linewidth=3
    # )
    # axs[0].plot(
    #     x[s], y2[s], "-o", c=cmap(0), label="(last epoch)", alpha=0.7, linewidth=3
    # )
    runs = reader.gather_runs(
        f"/checkpoint/rbalestriero/REVOLUTION2/BS_CV_LS_{dataset}_1/"
    )
    x = np.array([])
    y = np.array([])
    y2 = np.array([])
    z2 = np.array([])
    for run in runs:
        print(run["json_log"]["epoch"].iloc[-1])
        x = np.append(x, int(run["hparams"]["batch_size"]))
        y = np.append(y, run["json_log"]["accus"].iloc[:31].max() * 100)
        y2 = np.append(y2, run["json_log"]["accus"].max() * 100)
        z2 = np.append(
            z2, np.median(np.diff(run["json_log"]["relative_time"].iloc[2:].to_numpy()))
        )
    s = np.argsort(x)
    z = z2[s]
    axs[0].plot(
        x[s],
        y[s],
        "--o",
        c=cmap(1),
        label=r"lr:$\frac{0.001*256}{bs}$ (epoch30)",
        alpha=0.7,
        linewidth=3,
    )
    axs[0].plot(
        x[s], y2[s], "-o", c=cmap(1), label="(last epoch)", alpha=0.7, linewidth=3
    )
    axs[1].plot(x[s], z, "-o", c="k")
    axs[0].set_xscale("log")
    axs[1].set_yticks([int(z[0]), int(z[0] / 2 + z[-1] / 2), int(z[-1])])
    axs[1].set_xticks(
        [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        [str(i) for i in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]],
    )
    axs[1].xaxis.set_tick_params(rotation=90)
    axs[1].set_xlim(x[s[0]], x[s[-1]])

    legend = axs[0].legend(ncol=2)
    plt.setp(legend.get_title(), color="0.5")
    if "CIFAR" in dataset:
        axs[0].set_ylabel("top1 test acc. (%)")
        axs[1].set_ylabel("sec./epoch")
    axs[1].set_xlabel("batch size")
    axs[0].set_title(dataset, style="italic")
    fig.savefig(f"saved_BS_CV_{dataset}.png", dpi=100)
    plt.close()
    print(
        tabulate(
            np.round(np.stack([x[s], y2[s]]), 1),
            tablefmt="latex",
        )
    )


if __name__ == "__main__":
    # plot("CIFAR100")
    plot("TINYIMAGENET")
