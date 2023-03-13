from omega import reader
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import re
import pandas as pd
import json

cmap = plt.cm.jet
plt.rcParams.update(
    {
        "font.size": 14,
        # "figure.autolayout": True,
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


# fig, axs = reader.gather_group_plot(
#     "/checkpoint/rbalestriero/REVOLUTION3/ARCH_IMAGENET/",
#     group_by="architecture",
#     legend_by="max_indices",
#     file_key="log",
#     y_key="accus",
#     figsize=(16, 4),
#     sharex="all",
#     sharey="all",
# )
# fig.legend()
# plt.savefig(f"saved_CV_MANY.png")
# plt.close()
# asdf

# wd = "0.0"
import matplotlib.pyplot as plt

wd = ["0.05", "0.01"]
ls = ["0.2", "0.5", "0.7", "0.8", "0.9", "0.95", "0.99"]
fig, axs = plt.subplots(4, len(ls), sharex="all", sharey="row", figsize=(15, 15))
archs = ["densenet121", "resnet50", "convnext_tiny", "swin_t"]
data = reader.gather_all("/checkpoint/rbalestriero/DIET/label_smoothing_CV/IMAGENET100")
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
    return np.asarray(data["accus"] * 100)


df["accus"] = df.apply(row_reader, axis=1)


for i, smoothing in enumerate(ls):
    for j, arch in enumerate(archs):
        for proj in ["0", "2"]:
            selection = (
                (df["label_smoothing"] == smoothing)
                & (df["architecture"] == arch)
                & (df["projector_depth"] == proj)
            )
            subset = df.loc[selection]
            for t in range(len(subset)):
                axs[j, i].plot(subset.iloc[t]["accus"], c="k" if proj == "0" else "b")
for i, title in enumerate(archs):
    axs[i, 0].set_ylabel(title)
    axs[i, 0].set_ylim(30, 80)
for i, title in enumerate(ls):
    axs[0, i].set_title("ls:" + title)
plt.tight_layout()
plt.savefig("ls_CV.png")
plt.close()


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


archs = ["resnet18", "resnet50", "resnet101", "convnext_tiny", "swin_t"]
datasets = [
    "INATURALIST",
    "IMAGENET",
    "TINYIMAGENET",
    "IMAGENET100",
    "CIFAR100",
    "CIFAR10",
    "Food101",
]
all_data = []
all_data_std = []
to_plot = {}
table = np.zeros((len(archs), 3 * len(datasets), 2)) * np.nan
for j, dataset in enumerate(datasets):
    print(dataset)
    print("\tGathering")
    data = reader.gather_all(
        f"/checkpoint/rbalestriero/DIET/indices_LT_CV_V3/{dataset}"
    )
    df = pd.concat(data, axis=1).transpose()
    print("\tAccuing")
    df["accus"] = df.apply(row_reader, axis=1)
    df = df.groupby(["supervised", "max_indices", "architecture"])
    print("\tDone... Collecting")
    if dataset not in to_plot:
        to_plot[dataset] = {}
    for ind, ind_lbl in zip(
        ["500", "1000", "2000", "5000", "10000", "30000"],
        ["500", "1K", "2K", "5K", "10K", "30K"],
    ):
        for i, arch in enumerate(archs):
            if arch not in to_plot[dataset]:
                to_plot[dataset][arch] = []
            # table[i, j * 3, 0] = df.get_group(("True", ind, arch))["accus"].mean()
            # table[i, j * 3, 1] = df.get_group(("True", ind, arch))["accus"].std()
            to_plot[dataset][arch].append(
                df.get_group(("True", ind, arch))["accus"].max()
            )
            subset = df.get_group(("False", ind, arch))
            selection = (subset["label_smoothing"] == "0.8") & (
                subset["projector_depth"] == "0"
            )
            # table[i, j * 3 + 1, 0] = df.loc[selection]["accus"].mean()
            # table[i, j * 3 + 1, 1] = df.loc[selection]["accus"].std()
            to_plot[dataset][arch].append(subset.loc[selection]["accus"].max())
            selection = (subset["label_smoothing"] == "0.8") & (
                subset["projector_depth"].isin(["1", "2"])
            )
            # table[i, j * 3 + 2, 0] = df.loc[selection]["accus"].mean()
            # table[i, j * 3 + 2, 1] = df.loc[selection]["accus"].std()

    # final = np.round(100 * np.nanmax(data, -1), 1).astype("str")
    # final_std = np.round(100 * np.nanstd(data, -1), 1).astype("str")

    # # add arch info
    # final = np.concatenate([np.asarray(archs)[:, None], final], 1)
    # final_std = np.concatenate([np.asarray(archs)[:, None], final_std], 1)

    # # add index info
    # N = len(archs)
    # infos = ["\multirow{" + str(N) + "}{*}{" + ind_lbl + "}"] + [""] * (N - 1)
    # final = np.concatenate([np.asarray(infos)[:, None], final], 1)
    # final_std = np.concatenate([np.asarray(infos)[:, None], final_std], 1)
    # all_data.append(final)
    # all_data_std.append(final_std)

cmap = plt.cm.tab10
fig, axs = plt.subplots(
    3, len(archs), figsize=(4 * len(archs), 9), sharex="all", sharey="row"
)
for j, dataset in enumerate(["CIFAR100", "IMAGENET", "INATURALIST"]):
    for i, arch in enumerate(
        ["resnet18", "resnet50", "resnet101", "convnext_tiny", "swin_t"]
    ):
        axs[j, i].plot(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][::2],
            c="tab:red",
            linewidth=3,
            label="supervised" if i == 0 and j == 0 else "_None",
        )
        axs[j, i].scatter(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][::2],
            c="tab:red",
            edgecolors="k",
            linewidth=2,
        )
        axs[j, i].plot(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][1::2],
            c="tab:blue",
            linewidth=3,
            label="DIET" if i == 0 and j == 0 else "_None",
        )
        axs[j, i].scatter(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][1::2],
            c="tab:blue",
            edgecolors="k",
            linewidth=2,
        )
        axs[j, i].set_xlim(500, 30000)
        axs[j, i].set_xscale("log")
        axs[0, i].set_title(arch)
    axs[j, 0].set_ylabel(dataset)
axs[0, 0].set_xticks(
    [500, 1000, 2000, 5000, 10000, 30000], ["500", "1K", "2K", "5K", "10K", "30K"]
)
for i in range(axs.shape[1]):
    axs[-1, i].tick_params(axis="x", labelrotation=45)

axs[0, 0].legend(fontsize=20, loc="lower right")
axs[-1, 2].set_xlabel("train set size (N)")
plt.subplots_adjust(0.04, 0.12, 0.98, 0.96, 0.12, 0.05)
plt.savefig("ratio_plot_short.png")
plt.close()


fig, axs = plt.subplots(
    4, len(archs), figsize=(4 * len(archs), 12), sharex="all", sharey="row"
)
for j, dataset in enumerate(["CIFAR10", "TINYIMAGENET", "IMAGENET100", "Food101"]):
    for i, arch in enumerate(
        ["resnet18", "resnet50", "resnet101", "convnext_tiny", "swin_t"]
    ):
        axs[j, i].plot(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][::2],
            c="tab:red",
            linewidth=3,
            label="supervised" if i == 0 and j == 0 else "_None",
        )
        axs[j, i].scatter(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][::2],
            c="tab:red",
            edgecolors="k",
            linewidth=2,
        )
        axs[j, i].plot(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][1::2],
            c="tab:blue",
            linewidth=3,
            label="DIET" if i == 0 and j == 0 else "_None",
        )
        axs[j, i].scatter(
            [500, 1000, 2000, 5000, 10000, 30000],
            to_plot[dataset][arch][1::2],
            c="tab:blue",
            edgecolors="k",
            linewidth=2,
        )
        axs[j, i].set_xlim(500, 30000)
        axs[j, i].set_xscale("log")
        axs[0, i].set_title(arch)
    axs[j, 0].set_ylabel(dataset)
axs[0, 0].set_xticks(
    [500, 1000, 2000, 5000, 10000, 30000], ["500", "1K", "2K", "5K", "10K", "30K"]
)
for i in range(axs.shape[1]):
    axs[-1, i].tick_params(axis="x", labelrotation=45)
axs[0, 0].legend(fontsize=20, loc="lower right")
axs[-1, 2].set_xlabel("train set size (N)")
plt.subplots_adjust(0.04, 0.12, 0.98, 0.96, 0.12, 0.05)
plt.savefig("ratio_plot.png")
plt.close()

asdf
latex = tabulate(
    np.concatenate(all_data, 0),
    headers=["N", "arch"] + ["sup.", "DIET", "+proj"] * len(datasets),
    tablefmt="latex_raw",
).replace("nan", "-")
print(latex)
# set up lines
# pos = [m.start() for m in re.finditer(r"\\", latex)]
# offset = 0
# for p in pos[6::4]:
#     p += offset
#     print(p)
#     before = latex[:p]
#     after = latex[p:]
#     latex = before + after.replace(r"\\", r"\\\cline{2-19}", 1)
#     offset += len(r"\\\cline{2-19}") - 2


print(latex)
print(
    "\multicolumn{3}{c|}{} &"
    + "&".join(["\multicolumn{4}{c|}{" + dataset + "}" for dataset in datasets])
)
asasdf

# CIFAR 10
for i in ["10000", "50000"]:
    fig, axs, groups = reader.gather_group_plot(
        "/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/CIFAR10_CV/",
        column_key="architecture",
        row_key="projector_depth",
        file_key="eval_accuracies",
        color_by="loss",
        y_key="accus",
        figsize=(14, 8),
        sharex="all",
        sharey="all",
        filter={
            "supervised": "False",
            "max_indices": i,
        },
    )

    axs[0, 0].set_ylim(0.4, 0.9)
    axs[0, 0].set_yticks(
        np.linspace(0.4, 0.9, 6), [str(int(i * 100)) for i in np.linspace(0.4, 0.9, 6)]
    )
    axs[0, 0].set_ylabel("No projector")
    axs[1, 0].set_ylabel("1-layer projector")
    axs[0, 0].set_xlim(0, 4000)
    plt.savefig(f"saved_C10_{i}.png")
    plt.close()
    print(i)
    for key, runs in groups.items():
        m = 0
        for r in runs:
            m = max(m, r["eval_accuracies"]["accus"].max() * 100)
        print(key, m)


for dataset in ["TINYIMAGENET", "IMAGENET100"]:
    for arch in ["resnet18"]:
        baselines = reader.gather_group_plot(
            f"/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/{dataset}_CV/",
            column_key="max_indices",
            file_key="eval_accuracies",
            y_key="accus",
            filter={
                "supervised": "True",
                "architecture": arch,
                # "projector_depth": "0",
            },
        )[-1]
        DIET = reader.gather_group_plot(
            f"/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/{dataset}_CV/",
            column_key="max_indices",
            row_key="loss",
            file_key="eval_accuracies",
            y_key="accus",
            filter={
                "supervised": "False",
                "architecture": arch,
                # "projector_depth": "0",
            },
        )[-1]
        plt.close()

        for key, runs in baselines.items():
            m = 0
            for r in runs:
                m = max(m, r["eval_accuracies"]["accus"].max() * 100)
            print(key, m)
        for key, runs in DIET.items():
            m = 0
            for r in runs:
                m = max(m, r["eval_accuracies"]["accus"].max() * 100)
            print(key, m)


for arch in ["resnet18", "resnet50"]:
    for i in ["10000", "50000"]:
        fig, axs, groups = reader.gather_group_plot(
            "/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/CIFAR100_CV/",
            column_key="beta",
            color_by="label_smoothing",
            row_key="projector_depth",
            file_key="eval_accuracies",
            y_key="accus",
            figsize=(10, 10),
            sharex="all",
            sharey="all",
            filter={
                "loss": "boot",
                "architecture": arch,
                "max_indices": i,
            },
        )
        plt.savefig(f"saved_C100_{i}_{arch}_boot.png")
        plt.close()
        fig, axs, groups = reader.gather_group_plot(
            "/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/CIFAR100_CV/",
            column_key="loss",
            color_by="label_smoothing",
            row_key="projector_depth",
            file_key="eval_accuracies",
            y_key="accus",
            figsize=(10, 10),
            sharex="all",
            sharey="all",
            filter={
                "architecture": arch,
                "max_indices": i,
            },
        )
        plt.savefig(f"saved_C100_{i}_{arch}_all.png")
        plt.close()
        print(arch, i)
        for key, runs in groups.items():
            m = 0
            for r in runs:
                m = max(m, r["eval_accuracies"]["accus"].max() * 100)
            print(key, m)

# fig, axs, _ = reader.gather_group_plot(
#     "/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/IMAGENET_CV/",
#     column_key="beta",
#     color_by="label_smoothing",
#     row_key="projector_depth",
#     file_key="eval_accuracies",
#     y_key="accus",
#     figsize=(10, 10),
#     sharex="all",
#     sharey="all",
#     filter={
#         "loss": "boot",
#         "architecture": "resnet18",
#         "max_indices": "10000",
#     },
# )
# plt.savefig(f"saved_CV_MANY.png")
# plt.close()
# af
fig, axs, _ = reader.gather_group_plot(
    "/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/IMAGENET_CV/",
    column_key="loss",
    color_by="label_smoothing",
    row_key="projector_depth",
    file_key="eval_accuracies",
    y_key="accus",
    figsize=(10, 10),
    sharex="all",
    sharey="all",
    filter={
        "architecture": "resnet18",
        "max_indices": "10000",
    },
)
# for ax in axs:
#     ax.legend()
plt.savefig(f"saved_CV_MANY.png")
plt.close()
asdf
# /checkpoint/rbalestriero/REVOLUTION3/ARCH_INATURALIST/2c8cec7f-a5e0-4be8-b358-2b495e59fb79 (fast, overfit)
# --label-smoothing 0.95 --learning-rate 0.001 --weight-decay 0.05 --epochs 3000
# /checkpoint/rbalestriero/REVOLUTION3/ARCH_INATURALIST/e4f9b31e-dcd6-4d14-a75b-6b54eaf343ae (slow, no overfit)
# --label-smoothing 0.8 --learning-rate 0.001 --weight-decay 0.05 --epochs 1000


def plot(dataset, arch):
    runs = reader.gather_runs(f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_{dataset}/")
    accus = []
    values = []
    for run in runs:
        if run["hparams"]["architecture"] != arch:
            continue
        perc = run["log"]["epoch"].iloc[-1] / int(run["hparams"]["epochs"])
        values.append(
            (perc, run["log"]["accus"].max() * 100, run["hparams"]["max_indices"])
        )
        # if run["log"]["accus"].max() * 100 < 39:
        #     continue
        accus.append(
            (
                "k",
                100 * run["log"]["epoch"] / int(run["hparams"]["epochs"]),
                run["log"]["accus"].ewm(span=11).mean() * 100,
            )
        )

    return values, accus


if __name__ == "__main__":
    # runs = reader.gather_runs(f"/checkpoint/vivc/RB_experiments/ARCH_IMAGENET_GROUPED/")
    # runs += reader.gather_runs(
    #     f"/checkpoint/albertob/RB_experiments/ARCH_IMAGENET_GROUPED/"
    # )
    # runs += reader.gather_runs(
    #     f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_IMAGENET_GROUPED/"
    # )
    # for run in runs:
    #     perc = run["log"]["epoch"].iloc[-1] / int(run["hparams"]["epochs"])
    #     print(perc, run["log"]["accus"].max() * 100)
    archs = [
        "resnet18",
        "vit_b_16",
        "swin_s",
        "convnext_small",
    ]
    for dataset in ["INATURALIST", "IMAGENET"]:
        fig, axs = plt.subplots(
            1,
            4 + int(dataset == "IMAGENET"),
            sharey="all",
            sharex="all",
            figsize=(12, 4),
        )
        if dataset == "IMAGENET":
            archs += ["resnet50"]
        for i, arch in enumerate(archs):
            v, a = plot(dataset, arch)
            for (col, x, acc) in a:
                axs[i].plot(x, acc, c=col)
            axs[i].set_title(arch, style="italic")
            axs[i].set_xlabel("% training")
        axs[0].set_ylabel("top1 test acc. (%)")
        # print(a)

        plt.savefig(f"saved_{dataset}.png")
        plt.close()
        # print(tabulate(np.array(values).reshape((3, -1)).round(2), tablefmt="latex"))
