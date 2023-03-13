#!/bin/bash
module purge
module load anaconda3/2020.11-nold
source activate /private/home/rbalestriero/.conda/envs/ffcv

train_path="/private/home/rbalestriero/DATASETS/IMAGENET/train_500_jpg.ffcv"
val_path="/private/home/rbalestriero/DATASETS/IMAGENET/val_500_jpg.ffcv"
strength=3
epochs=2000

folder="/checkpoint/albertob/RB_experiments/ARCH_IMAGENET_GROUPED"
folder="/checkpoint/vivc/RB_experiments/ARCH_IMAGENET_GROUPED"
folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_IMAGENET_GROUPED"

bs=512
gpus=4
# from="/private/home/rbalestriero/noise-robust-vit/examples/in1k_randomproj_2048_grouped_10000.npz"

# from="/private/home/rbalestriero/noise-robust-vit/examples/in1k_randomproj_2048_grouped_100000.npz"

# lr="0.001"
# wd="0.05"
# for arch in resnet18 resnet50 convnext_small
# do
#     /private/home/rbalestriero/.conda/envs/ffcv/bin/python /private/home/rbalestriero/noise-robust-vit/examples/simpler_randomlabel.py --indices-from $from --sync-batchnorm --label-smoothing 0.8 --process-name GR$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
# done
# lr="0.0002"
# wd="0.01"
# for arch in swin_s vit_b_16
# do
#     /private/home/rbalestriero/.conda/envs/ffcv/bin/python /private/home/rbalestriero/noise-robust-vit/examples/simpler_randomlabel.py --indices-from $from --sync-batchnorm --label-smoothing 0.8 --process-name GR$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
# done
from="/private/home/rbalestriero/noise-robust-vit/examples/in1k_resnet18_2048_grouped_10000.npz"

lr="0.001"
wd="0.05"
for arch in resnet18 resnet50 convnext_small
do
    /private/home/rbalestriero/.conda/envs/ffcv/bin/python /private/home/rbalestriero/noise-robust-vit/examples/simpler_randomlabel.py --slurm-partition devlab --indices-from $from --sync-batchnorm --label-smoothing 0.8 --process-name GR$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
done
lr="0.0002"
wd="0.01"
for arch in swin_s vit_b_16
do
    /private/home/rbalestriero/.conda/envs/ffcv/bin/python /private/home/rbalestriero/noise-robust-vit/examples/simpler_randomlabel.py --slurm-partition devlab --indices-from $from --sync-batchnorm --label-smoothing 0.8 --process-name GR$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
done
