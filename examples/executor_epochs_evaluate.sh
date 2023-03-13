#!/bin/bash

gpus=1
strength=3
lr=0.001
wd=0.00001
epochs=100


arch=resnet50
export CUDA_VISIBLE_DEVICES=1

for name in StanfordCars #Food101 StanfordCars Flowers102 #CUB_200_2011 FGVCAircraft DTD OxfordIIITPet Food101 StanfordCars Flowers102
do
    train_path="/private/home/rbalestriero/DATASETS/${name}/train_jpg.ffcv"
    val_path="/private/home/rbalestriero/DATASETS/${name}/test_jpg.ffcv"
    # Resnet50 IN-1k
    checkpoint="/checkpoint/rbalestriero/REVOLUTION3/ARCH_IMAGENET/d7b6409f-11f0-4e24-8d60-8794e9510838/final.ckpt"
    folder="TODELETE_EVALUATION/R50_IN1k/${name}"
    python evaluation.py --checkpoint $checkpoint --process-name EVA$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    # Resnt50 Tiny
    checkpoint="/checkpoint/rbalestriero/REVOLUTION3/ARCH_TINYIMAGENET_3/0a901441-51ad-494c-8d1f-b014059bfc69/final.ckpt"
    folder="TODELETE_EVALUATION/R50_Tiny/${name}"
    python evaluation.py --checkpoint $checkpoint --process-name EVA$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    # Resnet50 IN100
    checkpoint="/checkpoint/rbalestriero/REVOLUTION3/ARCH_IMAGENET100_3/5d06f5a1-e34c-4548-9c6d-440ffda9988f/final.ckpt"
    folder="TODELETE_EVALUATION/R50_IN100/${name}"
    python evaluation.py --checkpoint $checkpoint --process-name EVA$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
done


