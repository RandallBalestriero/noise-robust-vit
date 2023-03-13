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

    print("\tGathering")
    data = reader.gather_all("/checkpoint/rbalestriero/DIET/LT_BIG/INATURALIST/")
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
    df = df.groupby(
        [
            "epochs",
            "wd_scaling",
            "architecture",
            "label_smoothing",
        ]
    )
    for g in df.groups.keys():
        print(g, df.get_group(g)["accus"].max())
