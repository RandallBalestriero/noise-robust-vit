from omega import reader
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

cmap = plt.cm.cool
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

col = {
    "50": (0, 0),
    "100": (0, 1),
    "200": (0, 2),
    "500": (1, 0),
    "1000": (1, 1),
    "5000": (1, 2),
}
colors = {
    "0.0": 0.1,
    "0.1": 0.2,
    "0.5": 0.3,
    "0.8": 0.4,
    "0.85": 0.5,
    "0.9": 0.6,
    "0.95": 0.7,
    "0.99": 0.9,
}


def plot(dataset, arch):
    runs = reader.gather_runs(
        f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_INATURALIST_CV_LS"
    )
    accus = []
    values = []
    for run in runs:
        perc = run["json_log"]["epoch"].iloc[-1] / int(run["hparams"]["epochs"])
        values.append(
            (perc, run["json_log"]["accus"].max() * 100, run["hparams"]["max_indices"])
        )
        accus.append(
            (
                cmap(np.sqrt(float(run["hparams"]["epochs"]))),
                100 * run["json_log"]["epoch"] / int(run["hparams"]["epochs"]),
                run["json_log"]["accus"].ewm(span=11).mean() * 100,
            )
        )

    return values, accus


if __name__ == "__main__":
    runs = reader.gather_runs(
        f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_INATURALIST_CV_LS"
    )

    fig, axs = plt.subplots(
        2,
        3,
        sharey="all",
        sharex="all",
        figsize=(12, 8),
    )
    for run in runs:
        # perc = run["json_log"]["epoch"] / int(run["hparams"]["epochs"])
        i, j = col[run["hparams"]["epochs"]]
        c = cmap(colors[run["hparams"]["label_smoothing"]])
        axs[i, j].plot(run["json_log"]["accus"] * 100, c=c, alpha=0.5)
        axs[i, j].set_xlabel("% training")
    axs[0, 0].set_ylabel("top1 test acc. (%)")
    axs[1, 0].set_ylabel("top1 test acc. (%)")

    plt.savefig(f"saved_SMARTINIT.png")
    plt.close()
