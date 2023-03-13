from omega import reader
from matplotlib import pyplot as plt
import numpy as np

loss_col = {
    "ce": 0,
    "sce": 1,
    "sboot": 2,
}

bs_col = {
    "256": 0,
    "512": 1,
    "1024": 2,
    "2048": 3,
}

wd_col = {
    "256": 0,
    "512": 1,
    "1024": 2,
    "2048": 3,
}
for da in [0, 3]:
    for path in [
        f"/checkpoint/rbalestriero/REVOLUTION/RANDOMFFCV_CIFAR10_{da}",
        # f"../../RANDOM_CORRECTED_{da}V2",
        # f"../../RANDOM_FOOD101_{da}V2",
        # f"../../RANDOM_AIRCRAFT_{da}V2",
        # f"../../RANDOM_CIFAR10_{da}V2",
        # f"../../RANDOM_CIFAR100_{da}V2",
        # f"../../RANDOM_OxfordIIITPet_{da}V2",
    ]:
        name = path.split("RANDOMFFCV_")[-1]
        runs = reader.gather_runs(path)

        fig, axs = plt.subplots(1, 4, sharey="all", figsize=(15, 5))

        x, y = reader.sensitivity_analysis(runs, "learning_rate", "accus")
        for i in range(len(x)):
            axs[0].scatter([float(x[i])] * len(y[i]), y[i], c="k")
        axs[0].set_title("learning rate")
        x, y = reader.sensitivity_analysis(runs, "weight_decay", "accus")
        for i in range(len(x)):
            axs[1].scatter([float(x[i])] * len(y[i]), y[i], c="k")
        axs[1].set_title("weight dec")
        x, y = reader.sensitivity_analysis(runs, "label_smoothing", "accus")
        for i in range(len(x)):
            axs[2].scatter([float(x[i])] * len(y[i]), y[i], c="k")
        axs[2].set_title("label smo")
        x, y = reader.sensitivity_analysis(runs, "proba", "accus")
        for i in range(len(x)):
            axs[3].scatter([float(x[i])] * len(y[i]), y[i], c="k")
        axs[3].set_title("proba")
        fig.tight_layout()
        fig.savefig(f"saved_{name.lower()}.png")
        plt.close()
