#!/bin/bash

train_path="/private/home/rbalestriero/DATASETS/IMAGENET/train_500_jpg.ffcv"
val_path="/private/home/rbalestriero/DATASETS/IMAGENET/val_500_jpg.ffcv"


# train_path="/private/home/rbalestriero/DATASETS/INATURALIST/train_500_0.50_90_0.ffcv"
# val_path="/private/home/rbalestriero/DATASETS/INATURALIST/val_500_0.50_90_0.ffcv"
strength=3
epochs=30
folder="/checkpoint/rbalestriero/REVOLUTION3/MANY"
bs=1024
gpus=4
lr="0.002"
wd="0.0001"
ls=0.1
for arch in "resnet18"
do
    paths="/checkpoint/rbalestriero/REVOLUTION3/MULTI_METHODS/IMAGENET/${arch}/*/final.ckpt"
    python many_to_ffcv_dataset.py --slurm-partition devlab --max-num-models 4 --path-to-models "${paths}" --label-smoothing $ls --process-name UNI$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
    python many_to_ffcv_dataset.py --slurm-partition devlab --max-num-models 8 --path-to-models "${paths}" --label-smoothing $ls --process-name UNI$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
    python many_to_ffcv_dataset.py --slurm-partition devlab --max-num-models 16 --path-to-models "${paths}" --label-smoothing $ls --process-name UNI$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
done
