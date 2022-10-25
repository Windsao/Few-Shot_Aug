#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=/data1/sw99/NOAH/experiments/LoRA/ViT-B_prompt_lora_8.yaml
CKPT=/data1/sw99/ViT-B_16.npz
WEIGHT_DECAY=0.0001


mkdir -p logs_noaug
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for LR in 0.005
do 
    for DATASET in oxford_pets oxford_flowers
    do
        for SEED in 2
        do
            for SHOT in 8 16
            do 
                CUDA_VISIBLE_DEVICES=6 python supernet_train_prompt.py --no_aug --data-path=./data/dataset/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_LoRA --batch-size=64 --lr=${LR} --epochs=100 --is_LoRA --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="none"\
                    2>&1 | tee -a logs_noaug/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-LoRA.log
          done
        done
    done
done
