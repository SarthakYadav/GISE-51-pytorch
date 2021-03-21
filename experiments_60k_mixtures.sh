#!/bin/bash
EXP_DIR=$1
RUN_INDEX=$2

echo "EXP_DIR: ${EXP_DIR}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "Experiments will therefore be saved to ${EXP_DIR}/r${RUN_INDEX}"

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_18_lmdb.cfg \
        -e ${EXP_DIR}/r${RUN_INDEX}/mixup/resnet_18_lmdb_adam_64x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_34_lmdb.cfg \
         -e ${EXP_DIR}/r${RUN_INDEX}/mixup/resnet_34_lmdb_adam_64x4_adam_mixers \
         --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/cifar_resnet_44_lmdb.cfg \
         -e ${EXP_DIR}/r${RUN_INDEX}/mixup/cifar_resnet_44_lmdb_adam_64x4_adam_mixers \
         --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4

python train_gise_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/cifar_resnet_56_lmdb.cfg \
         -e ${EXP_DIR}/r${RUN_INDEX}/mixup/cifar_resnet_56_lmdb_adam_64x4_adam_mixers \
         --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4


echo "SCRIPT END!!!!"
