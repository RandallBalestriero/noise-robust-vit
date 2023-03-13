#!/bin/bash
module purge
module load anaconda3/2020.11-nold
source activate /private/home/rbalestriero/.conda/envs/ffcv

train_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"
epochs=3000
bs=256
gpus=1
lr="0.001"
wd="0.05"
for strength in 1 2 3
do
    for arch in resnet18 resnet34 resnet50 resnet101
    do
        folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_TINYIMAGENET_DA/${arch}/${strength}"
        /private/home/rbalestriero/.conda/envs/ffcv/bin/python /private/home/rbalestriero/noise-robust-vit/examples/simpler_randomlabel.py --label-smoothing 0.8 --process-name DA$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    done
done