import omega
import torch
import torchvision
import matplotlib.pyplot as plt

for strength in [1, 2, 3]:
    pipes = omega.transforms.ffcv_imagenet_train_dataset(
        device="cuda:0", dtype=torch.float16, strength=strength
    )
    images = []
    loader = omega.ffcv.train_reader(
        path="../../DATASETS/IMAGENET/val_500_jpg.ffcv",
        pipelines={"image": pipes[0], "label": pipes[1]},
        batch_size=1,
        world_size=1,
        indices=range(1),
    )
    for i in range(36):
        for x, y in loader:
            print(x.shape)
            images.append(x)
    images = torch.cat(images).float().cpu()
    print(images.shape)
    image = torchvision.utils.make_grid(
        images,
        normalize=True,
        scale_each=True,
        nrow=6,
    )
    plt.imshow(image.permute(1, 2, 0), aspect="auto")
    plt.savefig(f"ffcv_loader_{strength}v2.png")
    plt.close()
