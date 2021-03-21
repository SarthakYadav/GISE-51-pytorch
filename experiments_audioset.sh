#!/bin/bash

EXP_DIR=$1
RUN_INDEX=$2

echo "EXP_DIR: ${EXP_DIR}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "Experiments will therefore be saved to ${EXP_DIR}_[ft,scratch]/r${RUN_INDEX}"

python train_gise_mixtures.py --cfg_file ./cfgs_v2/audioset/resnet_18_lmdb.cfg \
        -e ${EXP_DIR}_ft/r{RUN_INDEX}/mixup/resnet_18_lmdb_adam_32x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4 --random_clip_size 1000

python train_gise_mixtures.py --cfg_file ./cfgs_v2/audioset/resnet_18_lmdb_scratch.cfg \
        -e ${EXP_DIR}_scratch/r{RUN_INDEX}/mixup/resnet_18_lmdb_adam_32x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4 --random_clip_size 1000

echo "SCRIPT END!!!!"
