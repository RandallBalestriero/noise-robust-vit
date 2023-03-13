
for model in CNN MLP
do
    for aug in crop
    do
        for lamb in 10.0
        do  
            rm -rf "ALBERTO/${model}/${aug}/${lamb}"
            python alberto.py --folder "ALBERTO/${model}/${aug}/${lamb}" --epochs 100 --augmentation $aug --lamb $lamb --embedding-dim 256 --learning-rate 0.001 --model $model
        done
    done
done