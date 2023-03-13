#!/bin/bash


strength=3
arch=resnet18
lr=0.001
wd=0.05
base=256

# train_path="/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv"
# val_path="/private/home/rbalestriero/DATASETS/CIFAR100/val_raw.ffcv"
# gpus=1
# epochs=4000
# folder="/checkpoint/rbalestriero/REVOLUTION2/BS_CV_CIFAR100_1"
# for bs in 32 64 128 256 512 1024 2048
# do
#     mult=$(bc<<<"scale=5;$bs/$base")
#     slr=$(bc<<<"scale=5;$lr*$mult")
#     python simpler_randomlabel.py --process-name CV1$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $slr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
# done
# folder="/checkpoint/rbalestriero/REVOLUTION2/BS_CV_CIFAR100_0"
# for bs in 32 64 128 256 512 1024 2048
# do
#     python simpler_randomlabel.py --process-name CV0$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
# done
def test(data_labels):
    un = torch.unique(data_labels)
    indices = un.view(-1,1).eq(data_labels).int().argmax(1)
    return indices

train_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"
gpus=1
epochs=2000
folder="/checkpoint/rbalestriero/REVOLUTION2/BS_CV_LS_TINYIMAGENET_1"
for bs in 8 16 32 64 128 256 512 1024 2048
do
    mult=$(bc<<<"scale=5;$bs/$base")
    slr=$(bc<<<"scale=5;$lr*$mult")
    python simpler_randomlabel.py --process-name CV1$arch --gpus-per-node $gpus --folder $folder --label-smoothing 0.8 --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $slr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
done
# folder="/checkpoint/rbalestriero/REVOLUTION2/BS_CV_TINYIMAGENET_0"
# for bs in 32 64 128 256 512 1024 2048
# do
#     python simpler_randomlabel.py --process-name CV0$arch --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
# done