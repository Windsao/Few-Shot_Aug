#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=/home/sw99/NOAH/experiments/VPT/ViT-B_prompt_vpt_5.yaml
CKPT=/home/sw99/ViT-B_16.npz
WEIGHT_DECAY=0.0001


mkdir -p logs_adv
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft 

for LR in 0.005
do 
    for DATASET in food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft  
    do
        for SEED in 2
        do
            for SHOT in 8 16
            do 
                CUDA_VISIBLE_DEVICES=2,3 python supernet_train_prompt.py --data-path=./data/dataset/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_VPT_5 --batch-size=64 --lr=${LR} --epochs=100 --is_visual_prompt_tuning --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="none"\
                    2>&1 | tee -a logs_adv/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-VPT_5.log
          done
        done
    done
done
