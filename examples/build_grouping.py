# import faiss
import omega
import torch
from tqdm import tqdm
import numpy as np
import faiss
import torchvision

pipes = omega.transforms.ffcv_imagenet_train_dataset(
    device="cuda:0", dtype=torch.float16, strength=0
)

loader = omega.ffcv.val_reader(
    path="../../DATASETS/IMAGENET/train_500_jpg.ffcv",
    pipelines={"image": pipes[0], "label": pipes[1]},
    batch_size=1024,
    world_size=1,
    num_workers=64,
)
D = 4096
print(len(loader.indices))
W = torch.randn(224 * 224 * 3, D).cuda().half()
W /= np.sqrt(W.size(0))
# model = torch.nn.DataParallel(torchvision.models.resnet50()).cuda().half()
# model.fc = torch.nn.Identity()
# model.eval()
embeds = []
with torch.no_grad():
    for x, y in tqdm(loader):
        # x = model(x)
        x = x.flatten(1)
        torch.nn.functional.normalize(x, out=x)
        # embeds.append(x.cpu().numpy())
        embeds.append((x @ W).cpu().numpy())

    x_train = np.concatenate(embeds, 0).astype(np.float32)

    for k in [300000]:
        kmeans = faiss.Kmeans(
            d=x_train.shape[1],
            k=k,
            niter=20,
            nredo=2,
            verbose=True,
            spherical=True,
            gpu=True,
        )
        kmeans.train(x_train)
        indices = kmeans.index.search(x_train, 1)[1].flatten()
        # centroids = kmeans.centroids[:100]
        centroids = kmeans.centroids[
            :100
        ]  # torch.from_numpy(kmeans.centroids[:100]).cuda()
        # # revert the random projection to see them in image space
        # centroids = torch.linalg.lstsq(W.float().T, centroids.T).solution.T
        # # and reshape as image shape
        # centroids = centroids.cpu().view(100, 3, 224, 224)
        # save
        np.savez(
            f"in1k_randomproj_{D}_grouped_{k}.npz", indices=indices, centroids=centroids
        )
