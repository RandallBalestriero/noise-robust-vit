#!/bin/bash

gpus=1
epochs=7000
strength=3

train_path="/private/home/rbalestriero/DATASETS/IMAGENET100/train_jpg.ffcv"
val_path="/private/home/rbalestriero/DATASETS/IMAGENET100/val_jpg.ffcv"
folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_IMAGENET100_3"
lr="0.001"
wd="0.05"
for arch in resnet18 resnet34 resnet50 resnet101 wide_resnet50_2 resnext50_32x4d densenet121 convnext_tiny convnext_small
do
    python simpler_randomlabel.py --label-smoothing 0.8 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
done
lr="0.0002"
wd="0.01"
for arch in  swin_t swin_s MLPMixer vit_b_16
do
    python simpler_randomlabel.py --label-smoothing 0.8 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
done

# train_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"
# val_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"
# folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_TINYIMAGENET_3"
# lr="0.001"
# wd="0.05"
# for arch in resnet18 resnet34 resnet50 resnet101 wide_resnet50_2 resnext50_32x4d densenet121 convnext_tiny convnext_small
# do
#     python simpler_randomlabel.py --label-smoothing 0.8 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
# done
# lr="0.0002"
# wd="0.01"
# for arch in  swin_t swin_s MLPMixer vit_b_16
# do
#     python simpler_randomlabel.py --label-smoothing 0.8 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
# done


# train_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"
# val_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"
# folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_TINYIMAGENET_3"
# lr="0.001"
# wd="0.05"
# arch=resnet18
# epochs=6000
# python simpler_randomlabel.py --slurm-partition devlab --label-smoothing 0.8 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path

# epochs=4000
# python simpler_randomlabel.py --slurm-partition devlab --label-smoothing 0.95 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path

# epochs=6000
# python simpler_randomlabel.py --slurm-partition devlab --label-smoothing 0.95 --process-name TINY$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
