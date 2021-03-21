#!/bin/bash

python train_classifier.py --cfg_file ./cfgs_v2/vggsound/efficientnet-b1_lmdb_scratch.cfg -e /mnt/Workspace/experiments_vggsound/vggsound_scratch/no_mixup/efficientnet-b1_ft_64_adam --fp16 --random_clip_size 500 --val_clip_size 1000

python train_classifier.py --cfg_file ./cfgs_v2/vggsound/efficientnet-b1_lmdb.cfg -e /mnt/Workspace/experiments_vggsound/vggsound_ft/no_mixup/efficientnet-b1_ft_64_adam --fp16 --random_clip_size 500 --val_clip_size 1000

python train_classifier.py --cfg_file ./cfgs_v2/vggsound/resnet_18_lmdb.cfg -e /mnt/Workspace/experiments_vggsound/vggsound_ft/no_mixup/resnet_18_ft_64_adam --fp16 --random_clip_size 500 --val_clip_size 1000
