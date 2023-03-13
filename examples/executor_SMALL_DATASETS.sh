#!/bin/bash

gpus=1
epochs=200
name=Flowers102
train_path="/private/home/rbalestriero/DATASETS/${name}/train_jpg.ffcv"
val_path="/private/home/rbalestriero/DATASETS/${name}/test_jpg.ffcv"
folder="/checkpoint/rbalestriero/REVOLUTION4/ARCH_BOOT_${name}_3"
lr="0.001"
for output_dim in 64 256 1024
do
    for temperature in 0.05 0.1 0.3 1 5 50
    do
        for wd in 0.0001 0.05
        do
            for depth in 0 1
            do
                for ls in 0.0 0.1 0.5 0.8
                do
                    for arch in resnet18 #resnet50 convnext_small
                    do
                        python simpler_randomlabel.py --clip-output-dim $output_dim --temperature $temperature --clip --projector-depth $depth --projector-width 1024 --slurm-partition scavenge --label-smoothing $ls --process-name PROJ --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength 3 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --checkpoint-frequency 100
                    done
                done
            done
        done
    done
done
# lr="0.0002"
# wd="0.01"
# for arch in swin_t
# do
#     python simpler_randomlabel.py --projector-depth $depth --loss ce --label-smoothing $ls --process-name $name$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 64 --learning-rate $lr --weight-decay $wd --strength 3 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --checkpoint-frequency 100
# done
# for ls in 0.8 0.95
# do
#     for depth in 0 1
#     do
#         for name in Flowers102 CUB_200_2011 FGVCAircraft DTD OxfordIIITPet StanfordCars #Food101
#         do
#             train_path="/private/home/rbalestriero/DATASETS/${name}/train_jpg.ffcv"
#             val_path="/private/home/rbalestriero/DATASETS/${name}/test_jpg.ffcv"
#             folder="/checkpoint/rbalestriero/REVOLUTION2/ARCH_BOOT_${name}_3"
#             lr="0.001"
#             wd="0.05"
#             for arch in resnet18 resnet50 convnext_small
#             do
#                 python simpler_randomlabel.py --projector-depth $depth --loss ce --label-smoothing $ls --process-name $name$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 64 --learning-rate $lr --weight-decay $wd --strength 3 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --checkpoint-frequency 100
#             done
#             lr="0.0002"
#             wd="0.01"
#             for arch in swin_t
#             do
#                 python simpler_randomlabel.py --projector-depth $depth --loss ce --label-smoothing $ls --process-name $name$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 64 --learning-rate $lr --weight-decay $wd --strength 3 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --checkpoint-frequency 100
#             done
#         done
#     done
# done