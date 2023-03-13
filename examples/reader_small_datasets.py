from omega import reader
import numpy as np
from tabulate import tabulate


def plot(dataset, arch, depth, loss):
    runs = reader.gather_runs(
        f"/checkpoint/rbalestriero/REVOLUTION2/ARCH_BOOT_{dataset}_3/",
        filter={
            "projector_depth": depth,
            "architecture": arch,
            "loss": loss,
            "label_smoothing": "0.8",
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
    runs = reader.gather_runs(
        f"/checkpoint/rbalestriero/REVOLUTION4/ARCH_BOOT_Flowers102_3/",
        # f"/checkpoint/rbalestriero/REVOLUTION3/ARCH_BOOT_Food101_3/",
        # filter={
        #     "projector_depth": depth,
        #     "architecture": arch,
        #     "loss": loss,
        #     "label_smoothing": "0.8",
        # },
        verbose=False,
    )
    values = []
    for run in runs:
        if "eval_accuracies" not in run:
            continue
        values.append(run["eval_accuracies"]["accus"].max() * 100)
    print(sorted(values))
    asdf
    datasets = [
        "FGVCAircraft",
        "DTD",
        "OxfordIIITPet",
        "Flowers102",
        "CUB_200_2011",
        "Food101",
        "StanfordCars",
    ]
    loss = "ce"
    for depth in ["0", "1"]:
        values = []
        for arch in ["resnet18", "resnet50", "convnext_small", "swin_t"]:
            for dataset in datasets:
                values.append(plot(dataset, arch, depth, loss))
        print(
            tabulate(
                np.array(values).reshape((4, -1)).round(2),
                tablefmt="latex",
                headers=datasets,
            )
        )
