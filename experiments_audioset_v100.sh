#!/bin/bash

EXP_DIR=$1
RUN_INDEX=$2

echo "EXP_DIR: ${EXP_DIR}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "Experiments will therefore be saved to ${EXP_DIR}_[ft,scratch]/r${RUN_INDEX}"

echo "Finetuning "
python train_gise_mixtures.py --cfg_file ./cfgs_v2/audioset/efficientnet-b1_lmdb.cfg -e ${EXP_DIR}_ft/r{RUN_INDEX}/mixup/efficientnet-b1_ft_32_adam_mixup --use_mixup --fp16 --random_clip_size 1000

python train_gise_mixtures.py --cfg_file ./cfgs_v2/audioset/efficientnet-b1_lmdb_scratch.cfg -e ${EXP_DIR}_scratch/r{RUN_INDEX}/mixup/efficientnet-b1_ft_32_adam_mixup --use_mixup --fp16 --random_clip_size 1000
