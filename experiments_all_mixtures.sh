#!/bin/bash
EXP_DIR=$1
RUN_INDEX=$2

echo "EXP_DIR: ${EXP_DIR}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "Experiments will therefore be saved to ${EXP_DIR}/r${RUN_INDEX}"

python train_all_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_18_lmdb.cfg \
        -e ${EXP_DIR}/r${RUN_INDEX}/mixup/resnet_18_lmdb_adam_64x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4 \
        --lmdbs_list "Datasets/gise_mixtures_60k/mixtures_lmdb/train_5k.lmdb"

echo "train_10k"
python train_all_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_18_lmdb.cfg \
        -e ${EXP_DIR}/r${RUN_INDEX}/mixup/resnet_18_lmdb_adam_64x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4 \
        --lmdbs_list "Datasets/gise_mixtures_60k/mixtures_lmdb/train_10k.lmdb"

echo "train_15k"
python train_all_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_18_lmdb.cfg \
        -e ${EXP_DIR}/r${RUN_INDEX}/mixup/resnet_18_lmdb_adam_64x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4 \
        --lmdbs_list "Datasets/gise_mixtures_60k/mixtures_lmdb/train_15k.lmdb"

echo "train_20k"
python train_all_mixtures.py --cfg_file ./cfgs_v2/gise_mixtures_60k/resnet_18_lmdb.cfg \
        -e ${EXP_DIR}/r${RUN_INDEX}/mixup/resnet_18_lmdb_adam_64x4_adam_mixers \
        --epochs 50 --use_mixup --gpus "0, 1, 2, 3" --num_workers 6 --prefetch_factor 4 \
        --lmdbs_list "Datasets/gise_mixtures_60k/mixtures_lmdb/train_20k.lmdb"
