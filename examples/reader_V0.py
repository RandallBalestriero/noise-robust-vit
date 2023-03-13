from omega import reader
from matplotlib import pyplot as plt
import numpy as np

depths = {"0": 0, "2": 1, "4": 2, "6": 3}
widths = {"32": 0, "128": 1, "512": 2, "2048": 3}
smoothings = {"0.0": 0, "0.01": 1, "0.1": 2, "0.2": 3}
probas = {"0.0": 0, "0.1": 1, "0.2": 2}
wds = {"0.0": 0, "0.001": 1, "0.05": 2}
loss_col = {
    "l1": 0,
    "l2": 1,
    "ce": 2,
    "sce": 3,
    "sboot": 4,
    "bce": 5,
}

runs = reader.gather_runs("../../RANDOM_CORRECTED/")

fig1, axs1 = plt.subplots(4, 6, sharex="all", sharey="all", figsize=(9, 6))
fig2, axs2 = plt.subplots(4, 6, sharex="all", sharey="all", figsize=(9, 6))
fig3, axs3 = plt.subplots(4, 6, sharex="all", sharey="all", figsize=(9, 6))
fig4, axs4 = plt.subplots(3, 6, sharex="all", sharey="all", figsize=(9, 6))
fig5, axs5 = plt.subplots(3, 6, sharex="all", sharey="all", figsize=(9, 6))
for r in runs:
    accus = np.asarray(r["json_log"]["accus"].tolist()).flatten()
    c = loss_col[r["hparams"]["loss"]]
    r1 = depths[r["hparams"]["projector_depth"]]
    r2 = widths[r["hparams"]["projector_width"]]
    r3 = smoothings[r["hparams"]["label_smoothing"]]
    r4 = probas[r["hparams"]["proba"]]
    r5 = wds[r["hparams"]["weight_decay"]]
    axs1[r1, c].plot(accus)
    axs1[r1, c].set_title(f"{r['hparams']['loss']}, {r['hparams']['projector_depth']}")
    axs2[r2, c].plot(accus)
    axs2[r2, c].set_title(f"{r['hparams']['loss']}, {r['hparams']['projector_width']}")
    axs3[r3, c].plot(accus)
    axs3[r3, c].set_title(f"{r['hparams']['loss']}, {r['hparams']['label_smoothing']}")
    axs4[r4, c].plot(accus)
    axs4[r4, c].set_title(f"{r['hparams']['loss']}, {r['hparams']['proba']}")
    axs5[r5, c].plot(accus)
    axs5[r5, c].set_title(f"{r['hparams']['loss']}, {r['hparams']['weight_decay']}")


fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig1.savefig("saved_random1.png")
fig2.savefig("saved_random2.png")
fig3.savefig("saved_random3.png")
fig4.savefig("saved_random4.png")
fig5.savefig("saved_random5.png")
plt.close()

# col = {"0.0": 0, "0.01": 1, "0.1": 2, "0.2": 3}

# runs = reader.gather_runs("../../NOWAK/crossval/")

# fig, axs = plt.subplots(4, 4, sharex="all", sharey="row", figsize=(10, 4))
# for r in runs:
#     strength = int(r["hparams"]["strength"])
#     std = col[r["hparams"]["noise_std"]]
#     imp = r["hparams"]["improved"].lower() == "true"
#     axs[strength, std].plot(r["json_log"]["accus"], c="red" if imp else "blue")
#     print(r["json_log"]["accus"].iloc[-1])

# plt.tight_layout()
# plt.savefig("saved_nowak.png")
# plt.close()
