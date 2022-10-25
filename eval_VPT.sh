#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=/data1/sw99/NOAH/experiments/VPT/ViT-B_prompt_vpt_5.yaml
WEIGHT_DECAY=0.0001


# mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft 

for LR in 0.005
do 
    for DATASET in oxford_pets
    do
        for SEED in 2
        do
            for SHOT in 8 16
            do 
                CUDA_VISIBLE_DEVICES=2 python supernet_train_prompt.py --data-path=./data/dataset/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --output_dir=./eval/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_VPT_5 --batch-size=64 --lr=${LR} --epochs=100 --is_visual_prompt_tuning --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="none" --eval\
                --resume ./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_VPT_5/checkpoint.pth --mode retrain
          done
        done
    done
done
