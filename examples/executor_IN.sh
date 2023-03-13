#!/bin/bash
dataset_path="/private/home/rbalestriero/DATASETS"
strength=3
train_path="${dataset_path}/IMAGENET100/train_jpg.ffcv"
val_path="${dataset_path}/IMAGENET100/val_jpg.ffcv"


s1=1.0
s2=1000.0
ls=0.9
lr=0.001
wd=0.05
arch=resnet50
epochs=500
# for dataset in "TINYIMAGENET" "CIFAR100" "CIFAR10"
# do
#     train_path="${dataset_path}/${dataset}/train_raw.ffcv"
#     val_path="${dataset_path}/${dataset}/val_raw.ffcv"
#     python simpler_randomlabel.py --slurm-partition scavenge --lr-scaling $s1 --wd-scaling $s2 --label-smoothing $ls --process-name LBFGS --gpus-per-node 1 --folder ALBERTO/YES --add-version --epochs $epochs --float16 --architecture $arch --batch-size 128 --learning-rate $lr --weight-decay $wd --strength 3 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
# done


train_path="${dataset_path}/CIFAR10/train_raw.ffcv"
val_path="${dataset_path}/CIFAR10/val_raw.ffcv"

arch=resnet18
epochs=100
# for lr in 0.01 0.001 0.0001
# do
#     for wd in 0.0001 #0.0001 0.05
#     do
#         for ls in 0.1 0.9
#         do
#             for s1 in 0.001 0.01 0.1 1.0 10 100
#             do
#                 for s2 in 10000 100000 1000000 #0.01 0.1 1 10 100 1000 10000 100000 1000000
#                 do
#                     python simpler_randomlabel.py --slurm-partition scavenge --lr-scaling $s1 --wd-scaling $s2 --label-smoothing $ls --process-name LBFGS --gpus-per-node 1 --folder ALBERTO/YES --add-version --epochs $epochs --float16 --architecture $arch --batch-size 128 --learning-rate $lr --weight-decay $wd --strength 3 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
#                 done
#             done
#         done
#     done
# done
# folder="/checkpoint/rbalestriero/DIET/ENTROPY_IMAGENET100/"
# gpus=1
# epochs=1000
# wd=0.05
# lr=0.001
# bs=256
# arch=resnet18
# for ind in 1000 10000 100000
# do
#     for ls in 0.0 0.1 0.8
#     do
#         for w in 1 0.1 0.01 0.001 0.0001
#         do
#             python simpler_randomlabel.py --max-indices $ind --entropy-propagation $w --projector-depth 0 --label-smoothing $ls --process-name PROP --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --checkpoint-frequency 100 --eval-each-epoch
#             python simpler_randomlabel.py --max-indices $ind --entropy-propagation $w --projector-depth 2 --label-smoothing $ls --process-name PROP --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --checkpoint-frequency 100 --eval-each-epoch
#         done
#     done
# done


# python simpler_randomlabel.py --slurm-partition devlab --projector-depth 2 --label-smoothing $ls --process-name VERY --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model --eval-each-epoch
# arch=resnet18
# python simpler_randomlabel.py --slurm-partition devlab --projector-depth 0 --label-smoothing $ls --process-name VERY --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model --eval-each-epoch
# python simpler_randomlabel.py --slurm-partition devlab --projector-depth 2 --label-smoothing $ls --process-name VERY --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model --eval-each-epoch

# folder="/checkpoint/rbalestriero/DIET/label_smoothing_CV/IMAGENET100/"

bs=128
gpus=1

# for epochs in 2000
# do
#     for depth in 0 2
#     do
#         for ls in 0.2 0.5 0.7 0.8 0.9 0.95 0.99
#         do
#             for arch in convnext_tiny swin_t densenet121 resnet50 
#             do
#                 if [[ "$arch" == "swin_t" ]]
#                 then
#                     lr="0.0002"
#                     wd="0.01"
#                 else
#                     lr="0.001"
#                     wd="0.05"
#                 fi
#                 python simpler_randomlabel.py --projector-depth $depth --label-smoothing $ls --process-name LS$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model --eval-each-epoch
#                 python simpler_randomlabel.py --projector-depth $depth --label-smoothing $ls --process-name LS$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay 0 --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model --eval-each-epoch
#             done
#         done
#     done
# done



dataset_path="/private/home/rbalestriero/DATASETS"
# epochs=50
# save_path="/checkpoint/rbalestriero/DIET/indices_CV_V3"
epochs=1000
save_path="/checkpoint/rbalestriero/DIET/indices_LT_CV_V3"
bs=128
gpus=1
strength=3
for seed in 0
do
    for mi in 500 1000 2000 5000 10000 30000
    do
        adj_epochs=$((epochs*50000/mi))
        for arch in resnet18 resnet50 resnet101 convnext_tiny swin_t
        do
            if [[ "$arch" == "swin_t" ]]
            then
                lr="0.0002"
                wd="0.01"
            else
                lr="0.001"
                wd="0.05"
            fi
            # train_path="${dataset_path}/IMAGENET100/train_jpg.ffcv"
            # val_path="${dataset_path}/IMAGENET100/val_jpg.ffcv"
            # folder="${save_path}/IMAGENET100/"
            # python simpler_randomlabel.py --supervised --max-indices $mi --indices-seed $seed --process-name $mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --checkpoint-frequency 100 --eval-each-epoch
            # for depth in 0 1
            # do
            #     python simpler_randomlabel.py --projector-depth $depth --max-indices $mi --indices-seed $seed --label-smoothing 0.8 --process-name $mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --checkpoint-frequency 100 --eval-each-epoch
            # done
            # train_path="${dataset_path}/Food101/train_jpg.ffcv"
            # val_path="${dataset_path}/Food101/test_jpg.ffcv"
            # folder="${save_path}/Food101/"
            # python simpler_randomlabel.py --supervised --max-indices $mi --indices-seed $seed --process-name $mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --checkpoint-frequency 100 --eval-each-epoch
            # for depth in 0 1
            # do
            #     python simpler_randomlabel.py --projector-depth $depth --max-indices $mi --indices-seed $seed --label-smoothing 0.8 --process-name $mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --checkpoint-frequency 100 --eval-each-epoch
            # done
            # train_path="${dataset_path}/IMAGENET/train_500_jpg.ffcv"
            # val_path="${dataset_path}/IMAGENET/val_500_jpg.ffcv"
            # folder="${save_path}/IMAGENET/"
            # python simpler_randomlabel.py --supervised --max-indices $mi --indices-seed $seed --process-name $mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
            # for depth in 0 1
            # do
            #     python simpler_randomlabel.py --projector-depth $depth --max-indices $mi --indices-seed $seed --label-smoothing 0.8 --process-name $mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
            # done
            # train_path="${dataset_path}/INATURALIST/train_500_0.50_90_0.ffcv"
            # val_path="${dataset_path}/INATURALIST/val_500_0.50_90_0.ffcv"
            # folder="${save_path}/INATURALIST/"
            # python simpler_randomlabel.py --slurm-partition scavenge --supervised --max-indices $mi --indices-seed $seed --process-name INAT$mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
            # for depth in 0 1
            # do
            #     python simpler_randomlabel.py --slurm-partition scavenge --projector-depth $depth --max-indices $mi --indices-seed $seed --label-smoothing 0.8 --process-name INAT$mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
            # done
            # for dataset in "CIFAR100" #"TINYIMAGENET" "CIFAR100" "CIFAR10"
            # do
            #     train_path="${dataset_path}/${dataset}/train_raw.ffcv"
            #     val_path="${dataset_path}/${dataset}/val_raw.ffcv"
            #     folder="${save_path}/${dataset}/"
            #     # python simpler_randomlabel.py --slurm-partition scavenge --supervised --max-indices $mi --indices-seed $seed --process-name $dataset$mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
            #     for depth in 0
            #     do
            #         python simpler_randomlabel.py --slurm-partition scavenge --projector-depth $depth --max-indices $mi --indices-seed $seed --label-smoothing 0.8 --process-name $dataset$mi$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $adj_epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
            #     done
            # done
        done
    done
done




dataset_path="/private/home/rbalestriero/DATASETS"
save_path="/checkpoint/rbalestriero/DIET/LT_BIG_V2"
bs=128
gpus=1
strength=3
for epochs in 100 500 1000
do
    for arch in resnet18 #resnet50 convnext_tiny swin_t
    do
        if [[ "$arch" == "swin_t" ]]
        then
            lr="0.0002"
            wd="0.01"
        else
            lr="0.001"
            wd="0.05"
        fi
        for depth in 0
        do
            for ls in 0.1 0.8
            do
                for scaling in 1.0 1000.0
                do
                    train_path="${dataset_path}/IMAGENET100/train_jpg.ffcv"
                    val_path="${dataset_path}/IMAGENET100/val_jpg.ffcv"
                    folder="${save_path}/IMAGENET100/"
                    python simpler_randomlabel.py --slurm-partition scavenge --projector-depth $depth --label-smoothing $ls --process-name LONG$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --wd-scaling $scaling
                    train_path="${dataset_path}/INATURALIST/train_500_0.50_90_0.ffcv"
                    val_path="${dataset_path}/INATURALIST/val_500_0.50_90_0.ffcv"
                    folder="${save_path}/INATURALIST/"
                    python simpler_randomlabel.py --slurm-partition scavenge --projector-depth $depth --label-smoothing $ls --process-name LONG$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --wd-scaling $scaling
                    train_path="${dataset_path}/IMAGENET/train_500_jpg.ffcv"
                    val_path="${dataset_path}/IMAGENET/val_500_jpg.ffcv"
                    folder="${save_path}/IMAGENET/"
                    python simpler_randomlabel.py --slurm-partition scavenge --projector-depth $depth --label-smoothing $ls --process-name LONG$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --wd-scaling $scaling
                    python simpler_randomlabel.py --slurm-partition scavenge --projector-depth $depth --label-smoothing $ls --process-name LONG$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch --wd-scaling $scaling --max-indices 300000
                done
            done
        done
    done
done
