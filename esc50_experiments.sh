#!/bin/bash

echo "Training ResNet-18 FT"
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/resnet_18_lmdb_mixtures_fold_1 --epochs 50 --num_workers 6 --fold_index 1 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/resnet_18_lmdb_mixtures_fold_2 --epochs 50 --num_workers 6 --fold_index 2 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/resnet_18_lmdb_mixtures_fold_3 --epochs 50 --num_workers 6 --fold_index 3 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/resnet_18_lmdb_mixtures_fold_4 --epochs 50 --num_workers 6 --fold_index 4 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/resnet_18_lmdb_mixtures_fold_5 --epochs 50 --num_workers 6 --fold_index 5 --fp16

python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/resnet_18_lmdb_mixtures_fold_1 --epochs 50 --num_workers 6 --fold_index 1 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/resnet_18_lmdb_mixtures_fold_2 --epochs 50 --num_workers 6 --fold_index 2 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/resnet_18_lmdb_mixtures_fold_3 --epochs 50 --num_workers 6 --fold_index 3 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/resnet_18_lmdb_mixtures_fold_4 --epochs 50 --num_workers 6 --fold_index 4 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/resnet_18_lmdb_mixtures_fold_5 --epochs 50 --num_workers 6 --fold_index 5 --fp16

python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/resnet_18_lmdb_mixtures_fold_1 --epochs 50 --num_workers 6 --fold_index 1 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/resnet_18_lmdb_mixtures_fold_2 --epochs 50 --num_workers 6 --fold_index 2 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/resnet_18_lmdb_mixtures_fold_3 --epochs 50 --num_workers 6 --fold_index 3 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/resnet_18_lmdb_mixtures_fold_4 --epochs 50 --num_workers 6 --fold_index 4 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/resnet_18_lmdb_mixtures_fold_5 --epochs 50 --num_workers 6 --fold_index 5 --fp16

echo "Training EfficientNet-B1 Ft"
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/efficientnet-b1_lmdb_mixtures_fold_1 --epochs 50 --num_workers 6 --fold_index 1 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/efficientnet-b1_lmdb_mixtures_fold_2 --epochs 50 --num_workers 6 --fold_index 2 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/efficientnet-b1_lmdb_mixtures_fold_3 --epochs 50 --num_workers 6 --fold_index 3 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/efficientnet-b1_lmdb_mixtures_fold_4 --epochs 50 --num_workers 6 --fold_index 4 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r1/efficientnet-b1_lmdb_mixtures_fold_5 --epochs 50 --num_workers 6 --fold_index 5 --fp16

python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/efficientnet-b1_lmdb_mixtures_fold_1 --epochs 50 --num_workers 6 --fold_index 1 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/efficientnet-b1_lmdb_mixtures_fold_2 --epochs 50 --num_workers 6 --fold_index 2 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/efficientnet-b1_lmdb_mixtures_fold_3 --epochs 50 --num_workers 6 --fold_index 3 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/efficientnet-b1_lmdb_mixtures_fold_4 --epochs 50 --num_workers 6 --fold_index 4 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r2/efficientnet-b1_lmdb_mixtures_fold_5 --epochs 50 --num_workers 6 --fold_index 5 --fp16

python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/efficientnet-b1_lmdb_mixtures_fold_1 --epochs 50 --num_workers 6 --fold_index 1 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/efficientnet-b1_lmdb_mixtures_fold_2 --epochs 50 --num_workers 6 --fold_index 2 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/efficientnet-b1_lmdb_mixtures_fold_3 --epochs 50 --num_workers 6 --fold_index 3 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/efficientnet-b1_lmdb_mixtures_fold_4 --epochs 50 --num_workers 6 --fold_index 4 --fp16
python train_esc50_classifier.py --cfg_file ./cfgs_v2/esc50/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_esc50/esc50_mixtures_v3_silence_ft_r3/efficientnet-b1_lmdb_mixtures_fold_5 --epochs 50 --num_workers 6 --fold_index 5 --fp16
