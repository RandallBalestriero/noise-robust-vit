from omega import reader
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


# get colormap
cmap = plt.cm.tab10
plt.rcParams.update(
    {
        "font.size": 14,
        "figure.autolayout": True,
        "figure.titlesize": 22,
        "figure.titleweight": "bold",
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


def plot(dataset, arch):
    values = {}
    accus = {}
    for strength in [1, 2, 3]:
        runs = reader.gather_runs(
            f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_{dataset}_DA/{arch}/{strength}"
        )
        values[strength] = []
        accus[strength] = []
        for run in runs:
            values[strength].append(run["json_log"]["accus"].ewm(span=15).mean() * 100)
            accus[strength].append(run["json_log"]["accus"].max() * 100)
        accus[
            strength
        ] = f"{np.round(np.mean(accus[strength]),2)}$\\pm$ {np.round(np.std(accus[strength]),1)}"
    return values, accus


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, sharex="all", figsize=(10, 10))

    dataset = "TINYIMAGENET"
    accus = []
    for m, arch in enumerate(
        [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ]
    ):
        values, accs = plot(dataset, arch)
        accus.append(list(accs.values()))
        for i in [1, 2, 3]:
            doit = True
            for v in values[i]:
                if m == 0 and doit:
                    label = f"DA strength {i}"
                    doit = False
                else:
                    label = "_None"
                axs[m // 2, m % 2].plot(
                    v, linewidth=2, alpha=0.5, color=cmap(i - 1), label=label
                )
            axs[m // 2, m % 2].set_xlim([0, 3000])
            axs[m // 2, m % 2].set_ylim([20, values[2][0].max() + 5])
            axs[m // 2, m % 2].set_title(arch, style="italic")
    axs[0, 0].legend()
    axs[1, 0].set_xlabel("epochs")
    axs[1, 1].set_xlabel("epochs")
    axs[0, 0].set_ylabel("top1 test acc (%)")
    axs[1, 0].set_ylabel("top1 test acc (%)")
    fig.suptitle("TinyImagenet", style="italic")
    plt.savefig("saved_DA.png")

    print(
        tabulate(
            np.asarray(accus).reshape((4, 3)),
            tablefmt="latex_raw",
            headers=["strength: 1", "strength: 2", "strength: 3"],
        )
    )
    # print(tabulate(np.array(values).reshape((3, -1)).round(2), tablefmt="latex"))
