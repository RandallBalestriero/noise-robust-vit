import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(21, 7))
N = 40
G = np.zeros((N, N))
G[: N // 2, : N // 2] = 1
G[N // 2 :, N // 2 :] = 1
im = axs[0].imshow(
    G, interpolation="none", vmin=0, vmax=1, aspect="equal", cmap="Greys"
)
G = np.eye(N)
G[: N // 4, : N // 4] = 1
G[N // 2 : N // 4 + N // 2, N // 2 : N // 4 + N // 2] = 1
im = axs[1].imshow(
    G, interpolation="none", vmin=0, vmax=1, aspect="equal", cmap="Greys"
)
G = np.eye(N)
for i in range(100):
    a = np.random.randint(N)
    b = np.random.randint(N)
    if a < N // 2 and b < N // 2:
        G[a, b] = 1
    elif a > N // 2 and b > N // 2:
        G[a, b] = 1
im = axs[2].imshow(
    G, interpolation="none", vmin=0, vmax=1, aspect="equal", cmap="Greys"
)


for i in range(3):
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].set_xticks(np.arange(-0.5, G.shape[1], 1), minor=True)
    axs[i].set_yticks(np.arange(-0.5, G.shape[0], 1), minor=True)
    axs[i].grid(which="minor", color="grey", linestyle="-", linewidth=1)
    axs[i].tick_params(which="minor", bottom=False, left=False)


plt.tight_layout()
plt.savefig("G_matrix.png")
plt.close()
