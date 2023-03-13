from omega import reader
from matplotlib import pyplot as plt
import numpy as np


for da in ["", "DA_"]:
    for path in [
        f"../../1000_CORRECTED_{da}V2",
        # f"../../1000_FOOD101_{da}V2",
        # f"../../1000_AIRCRAFT_{da}V2",
        # f"../../1000_CIFAR10_{da}V2",
        # f"../../1000_CIFAR100_{da}V2",
        # f"../../1000_OxfordIIITPet_{da}V2",
    ]:
        name = path.split("1000_")[-1]
        runs = reader.gather_runs(path)
        fig, axs = plt.subplots(1, 2, sharey="all", figsize=(15, 5))

        for r in runs:
            # if int(r["hparams"]["projector_depth"]) > 0:
            #     continue
            if r["hparams"]["dataset_path"] != "/datasets01/imagenet_full_size/061417":
                continue
            accus = np.asarray(r["json_log"]["accus"].tolist()).flatten()
            if r["hparams"]["architecture"] == "resnet50":
                col = 1
            else:
                col = 0
            axs[col].plot(accus)

        fig.tight_layout()
        fig.savefig(f"saved1000_{name.lower()}.png")
        plt.close()
