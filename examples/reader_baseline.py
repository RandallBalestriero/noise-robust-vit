from omega import reader
from matplotlib import pyplot as plt
import numpy as np

data_col = {
    "/private/home/rbalestriero/DATASETS/Food101": (0, 0),
    "/private/home/rbalestriero/DATASETS/Flowers102": (0, 1),
    "/private/home/rbalestriero/DATASETS/FGVCAircraft": (0, 2),
    "/private/home/rbalestriero/DATASETS/OxfordIIITPet": (1, 0),
    "/private/home/rbalestriero/DATASETS/DTD": (1, 1),
    "/private/home/rbalestriero/DATASETS/Country211": (1, 2),
}

runs = reader.gather_runs("../../BASELINES/test")

fig, axs = plt.subplots(2, 3, figsize=(15, 5))

for r in runs:
    accus = np.asarray(r["json_log"]["accus"].tolist()).flatten()
    c = data_col[r["hparams"]["dataset_path"]]
    axs[c[0], c[1]].plot(accus)
    print(r["hparams"]["dataset_path"], accus.max())
    axs[c[0], c[1]].set_title(f"{r['hparams']['dataset_path'].split('/')[-1]}")

fig.tight_layout()
fig.savefig("saved_baseline.png")
plt.close()
