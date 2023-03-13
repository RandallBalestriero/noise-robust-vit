from omega import reader
import numpy as np
import os

n_runs = 1
for da in ["", "DA_"]:
    for path in [
        f"../../RANDOM_CORRECTED_{da}V2",
        # f"../../RANDOM_FOOD101_{da}V2",
        # f"../../RANDOM_AIRCRAFT_{da}V2",
        # f"../../RANDOM_CIFAR10_{da}V2",
        # f"../../RANDOM_CIFAR100_{da}V2",
        # f"../../RANDOM_OxfordIIITPet_{da}V2",
    ]:
        name = path.split("RANDOM_")[-1]
        runs = reader.gather_runs(path)

        values = []
        configs = []

        for r in runs:
            accus = np.asarray(r["json_log"]["accus"].tolist()).flatten()
            # if int(r["hparams"]["projector_depth"]) > 0:
            #     continue
            values.append(accus.max())
            configs.append(r["hparams"])

        ordering = np.argsort(values)[-n_runs:]
        values = [values[i] for i in ordering]
        configs = [configs[i] for i in ordering]

        print(values)
        for r in configs:
            for architecture in ["resnet18", "resnet50"][-1:]:
                label_smoothing = r["label_smoothing"]
                optimizer = r["optimizer"]
                weight_decay = r["weight_decay"]
                loss = r["loss"]
                proba = r["proba"]
                learning_rate = r["learning_rate"]
                beta = r["beta"]
                projector_depth = r["projector_depth"]
                projector_width = r["projector_width"]
                dataset_path = r["dataset_path"]
                if "tiny" in dataset_path:
                    dataset_paths = [
                        # "/datasets01/tinyimagenet/081318/",
                        "/datasets01/imagenet_full_size/061417/",
                    ]
                else:
                    dataset_paths = [dataset_path]
                for dataset_path in dataset_paths:
                    if "imagenet_full_size" in dataset_path:
                        batch_size = 1024
                    else:
                        batch_size = r["batch_size"]
                    strength = r["strength"]
                    command = f"python randomlabel.py --gpus-per-node 8 --process-name long{architecture} --timeout-min 4200 --folder {path.replace('RANDOM', '1000')}/{architecture} --sync-batchnorm --add-version --float16 --epochs 1000 --dataset-path {dataset_path} --architecture {architecture} --label-smoothing {label_smoothing} --optimizer {optimizer} --weight-decay {weight_decay} --batch-size {batch_size} --learning-rate {learning_rate} --strength {strength} --save-final-model --loss {loss} --proba {proba} --beta {beta} --projector-depth {projector_depth} --projector-width {projector_width}"
                    print(command)
                    os.system(command)
