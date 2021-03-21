#!/bin/bash

EXP_DIR=$1
RUN_INDEX=$2

echo "EXP_DIR: ${EXP_DIR}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "Experiments will therefore be saved to ${EXP_DIR}/r${RUN_INDEX}"

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_50_lmdb.cfg -e ${EXP}/r${RUN_INDEX}/mixup/resnet50_scratch_64_adam_mixup --use_mixup --fp16

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/efficientnet-b0_lmdb.cfg -e ${EXP}/r${RUN_INDEX}/mixup/efficientnet-b0_scratch_128_adam_mixup --use_mixup --fp16

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/efficientnet-b1_lmdb.cfg -e ${EXP}/r${RUN_INDEX}/mixup/efficientnet-b1_scratch_64_adam_mixup --use_mixup --fp16

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/densenet_121_lmdb.cfg -e ${EXP}/r${RUN_INDEX}/mixup/densenet121_scratch_64_adam_mixup --use_mixup --fp16
