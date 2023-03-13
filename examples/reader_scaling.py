from omega import reader
import numpy as np
from tabulate import tabulate
import pandas as pd
import json
import matplotlib.pyplot as plt

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
        "grid.color": "0.5",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
    }
)


def plot(ls, lr, s1, s2, arch, wd):

    runs = reader.gather_runs(
        f"ALBERTO/YES",
        filter={
            "lr_scaling": str(s1),
            "wd_scaling": str(s2),
            "architecture": arch,
            "label_smoothing": str(ls),
            "learning_rate": str(lr),
            "weight_decay": str(wd),
        },
        verbose=False,
    )
    values = []
    for run in runs:
        if "eval_accuracies" not in run:
            continue

        values.append(run["eval_accuracies"]["accus"].max() * 100)
    if len(values) == 0:
        return np.nan
    return np.max(values)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    S1 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    S2 = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

    print("\tGathering")
    data = reader.gather_all("ALBERTO/YES")
    df = pd.concat(data, axis=1).transpose()

    def row_reader(row):
        log = row["path"] / "eval_accuracies.txt"
        if not log.is_file():
            return np.nan
        lines = []
        try:
            with open(log, "r") as f:
                for line in f:
                    lines.append(json.loads(line))
        except json.decoder.JSONDecodeError:
            print(f"error reading last line of file {log}")
        data = pd.DataFrame(lines)
        return data["accus"].max() * 100

    df["accus"] = df.apply(row_reader, axis=1)
    best = df["accus"].argmax()
    print(
        df.iloc[best][
            [
                "lr_scaling",
                "wd_scaling",
                "label_smoothing",
                "learning_rate",
                "weight_decay",
            ]
        ]
    )
    df = df.groupby(
        [
            "lr_scaling",
            "wd_scaling",
            "architecture",
            "label_smoothing",
            "learning_rate",
            "weight_decay",
        ]
    )

    fig, axs = plt.subplots(4, 3, sharex="all", sharey="all", figsize=(15, 20))
    for k, (wd, wd_label) in enumerate(zip(["0.05", "0.0001"], ["5e-2", "1e-4"])):
        for i, ls in enumerate([0.1, 0.9]):
            for j, lr in enumerate([0.01, 0.001, 0.0001]):
                values = np.zeros((len(S1), len(S2)))
                for a, s1 in enumerate(S1):
                    for b, s2 in enumerate(S2):
                        try:
                            values[a, b] = df.get_group(
                                (
                                    str(s1),
                                    str(s2),
                                    "resnet18",
                                    str(ls),
                                    str(lr),
                                    str(wd),
                                )
                            )["accus"].max()
                        except KeyError:
                            continue
                print("\tplotting")
                axs[k * 2 + i, j].imshow(
                    values.T, interpolation="nearest", aspect="auto", origin="lower"
                )
                values = np.round(values, 1)
                for a in range(len(S1)):
                    for b in range(len(S2)):
                        axs[k * 2 + i, j].text(
                            a,
                            b,
                            str(values[a, b]),
                            c="r",
                            horizontalalignment="center",
                            verticalalignment="center",
                        )
                axs[k * 2 + i, j].set_title(f"ls:{ls}, lr:{lr}, wd:{wd_label}")

    axs[0, 0].set_xticks(range(len(S1)), S1)
    axs[0, 0].set_yticks(range(len(S2)), S2)
    for i in range(axs.shape[0]):
        axs[i, 0].set_ylabel("wd scaling")
    for i in range(axs.shape[1]):
        axs[-1, i].set_xlabel("lr scaling")
    plt.savefig(f"scaling_figure.png")
    plt.close()

    # print(
    #     tabulate(
    #         np.array(values).reshape((4, -1)).round(2),
    #         tablefmt="latex",
    #         headers=datasets,
    #     )
    # )
