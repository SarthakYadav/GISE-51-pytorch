# GISE-51-pytorch

Official code release for the paper [GISE-51: A scalable isolated sound events dataset](https://arxiv.org/abs/2103.12306) in pytorch using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning). Instructions and implementation for replicating all baseline experiments, including pre-trained models. If you use this code or part of it, please cite:
> Sarthak Yadav and Mary Ellen Foster, "GISE-51: A scalable isolated sound events dataset", arXiv:2103.12306, 2021

## About
Most of the existing isolated sound event datasets comprise a small number of sound event classes,  usually 10 to 15,  restricted to a small domain, such as domestic and urban sound events. In this work, we introduce GISE-51, a dataset derived from [FSD50K](https://arxiv.org/abs/2010.00475), spanning 51 isolated sound event classes belonging to a broad domain of event types. We also release GISE-51-Mixtures, a dataset of 5-second soundscapes with hard-labelled event boundaries synthesized from GISE-51 isolated sound events. We conduct baseline sound event recognition (SER) experiments on the GISE-51-Mixtures dataset, benchmarking prominent convolutional neural networks, and models trained with the dataset demonstrate strong transfer learning performance on existing audio recognition benchmarks. Together, GISE-51 and GISE-51-Mixtures attempt to address some of the shortcomings of recent sound event datasets, providing an open, reproducible benchmark for future research along with the freedom to adapt the included isolated sound events for domain-specific applications.

This repository contains code for reproducing all experiments done in the paper. For more information on the dataset, as well as the pre-trained models, visit [GISE-51: A scalable isolated sound events dataset](https://zenodo.org/record/4593514#.YFrRsEMzZhE)

## Keypoints
### In-memory data loading
* The implementation utilizes `lmdb` format for dataset serialization, and loads data in-memory while training (`src/data/inmemory_lmdb_dataset.py`). To keep memory usage to a minimum, the records in the lmdb's are not numpy arrays; they're entire flac file bytes written into lmdb's. For more info view `pack_mixtures_into_lmdb.py`.
* This was done because majority of the experiments were done on resource with slow I/O performance.  This had the added benefit of great GPU utilization, and works well for our dataset use cases. 
* One can easily modify `src/data/inmemory_lmdb_dataset.py` if you need a regular dataset that reads data from disk on-the-fly.

### Pre-trained models
As mentioned in the paper, we performed experiments several times and reported average results. However, since releasing so many models would be cumbersome, only the checkpoints from best performing run for an experiment are released. If you need checkpoints from all the runs, please get in touch.

### Per dataset configuration files
We utilize `.cfg` based training. cfgs for each CNN architecture used for each dataset are provided in `cfgs_v2`. These cfg files list hyperparameters for each experiment, and paired with provided `experiments_*.sh` files give exact settings the experiment was run under.

### Auxiliary datasets
Apart from the proposed dataset, we conduct experiments on 3 other datasets, viz. AudioSet _balanced_ [], VGGSound[] and ESC-50 []. Since AudioSet and VGGSound are based on YouTube videos, we cannot provide them to aid replication. However, we do provide list of youtube video ids for Audioset that we used in our experiments. Since ESC-50 is CC licensed, we can provide lmdb files for it.

## Requirements
* `torch==1.7.1` and corresponding `torchaudio` from [official pytorch](https://pytorch.org/get-started/locally/)
* `libsndfile-dev` from OS repos, for 'SoundFile==0.10.3'
* `requirements.txt`

## Procedure for replicating experiments

### Generating GISE-51-Mixtures data (not required)
To generate GISE-51-Mixtures, `scripts/generate_mixtures.py` was used. It's provided as a good starting point for generating data using Scaper. However, we provide exact mixtures used by us as a part of the [release](), and you don't need to generate mixtures from scratch again.

If you do want to generate exact mixtures from `isolated_events.tar.gz`, you'll have to use the provided `mixtures_jams.tar.gz`, which contains `.jams` annotation files and can be used by Scaper to generate mixtures from. For more information, visit [Scaper Tutorials](https://scaper.readthedocs.io/en/latest/tutorial.html).

### Data Preparation
Before commencing with experiments, lmdb files need to be created. There are two options for generating lmdb files:
1. Downloading provided tar archives and pack them into lmdbs. <b>(Recommended)</b>
2. Downloading `noises.tar.gz`, `isolated_events.tar.gz` and `mixtures_jams.tar.gz`, generate soundscapes locally using scaper from jams files, convert soundscapes to `.flac` (for lower memory usage) using `scripts/wav2flac.py` and pack them into mixtures.

We recommend using method 1. For packing soundscape mixtures into lmdbs:

* Download `train*.tar.gz` (all if you want to repeat 4.1, just `train.tar.gz` otherwise), `val.tar.gz` and `eval.tar.gz` from [dataset page](https://zenodo.org/record/4593514#.YFrRsEMzZhE) and extract into a separate directory.
* Run `pack_mixtures_into_lmdb.py` to generate lmdb files from soundscape mixtures as follows. 
   ```
    # for generating lmdb for 60k soundscapes
    python pack_mixtures_into_lmdb.py --mixture_dir "mixtures_flac/train" --lmdb_path mixtures_lmdb/train.lmdb --map_size 2e10

    # for generating val lmdb
    python pack_mixtures_into_lmdb.py --mixture_dir "mixtures_flac/val" --lmdb_path mixtures_lmdb/val.lmdb --map_size 2e10

    # for generating eval lmdb
    python pack_mixtures_into_lmdb.py --mixture_dir "mixtures_flac/eval" --lmdb_path mixtures_lmdb/eval.lmdb --map_size 2e10
   ```
* `prepare_mixtures_lmdb.sh` can be used to create all the GISE-51-Mixtures lmdb files and as reference for creating specific lmdb files.

```
./prepare_mixtures_lmdb.sh <MIXTURES_DIR> <LMDB_DIR>
```

* lmdbs for AudioSet, VGGSound and ESC-50 experiments can be generated in a similar manner. ESC-50 lmdbs can be found [here](https://gla-my.sharepoint.com/:f:/g/personal/2552300y_student_gla_ac_uk/EpZkbOWaqvtHkoJJGFUq774BlDcQwj5s-nMhf3vBVjvBOQ?e=PotHV9).

### Experiments
The following sections outline how to reproduce experiments conducted in the paper. Before running any experiment, make sure your paths are correct in the corresponding `.cfg` files. Pre-trained models can be found [here](https://zenodo.org/record/4593514#.YFrRsEMzZhE). Download and extract the `pretrained-models.tar.gz` file.

`lbl_map.json` contains a json serialized dictionary that maps dataset labels to corresponding integer indices. To allow inference on pre-trained models, we provide `lbl_map.json` corresponding to all datasets in the `./data` directory.

---
#### Number of synthesized soundscapes v/s val mAP
This section studies how val mAP performance scales with number of training mixtures, using ResNet-18 architecture.  To prepare lmdb files for this experiment, 

To run this experiment, simply use `train_all_mixtures.py` as below
```
python train_all_mixtures.py --cfg_file ./cfgs_v2/v3_silence_trimmed/resnet_18_lmdb.cfg \
        -e <EXPERIMENT_DIR> --epochs 50 --use_mixup --gpus "0, 1, 2, 3" \
        --num_workers 6 --prefetch_factor 4 \
        --lmdbs_list "mixtures_lmdb/train_5k.lmdb;mixtures_lmdb/train_10k.lmdb;mixtures_lmdb/train_15k.lmdb;..."
```
where lmdbs_list is a `;` separated list of lmdb files. The script will run experiments looping over the provided lmdbs, saving checkpoints and logs to subfolders in <EXPERIMENT_DIR>
A sample script used is provided (`experiments_all_mixtures.sh`) and can be modified to train for all settings.

```
./experiments_all_mixtures.sh <EXPERIMENT_DIR> <RUN_INDEX>
```

Once this is done, you need to run evaluation to check performance on the *eval* set. Make sure to filter only the best checkpoints, and then run
```
python eval_all_mixtures.py --lbl_map meta/lbl_map.json --lmdb_path mixtures_lmdb/eval.lmdb --exp_dir <EXPERIMENT_DIR>/r<RUN_INDEX> --results_csv <EXPERIMENT_DIR>/r<RUN_INDEX>/results.csv
```
This will write evaluation metrics for <RUN_INDEX> in `results.csv`. One can then average results across runs. 

The pretrained models for this experiment are not provided, as there would be too many checkpoints.

---
#### CNN Baselines
This experiment studies performance of various CNN architectures trained on 60k synthetic soundscapes. The experiments were run on two different machines, with ResNet-50, EfficientNet-B0/B1 and DenseNet-121 ran on single V100 machine, and all other experiments run on 4x GTX 1080 machine.
Scripts `experiments_60k_mixtures.sh` and `experiments_60k_mixtures_v100.sh` were used for training these models as follows:
```
./experiments_60k_mixtures.sh <EXPERIMENT_DIR> <RUN_INDEX>
./experiments_60k_mixtures_v100.sh <EXPERIMENT_DIR> <RUN_INDEX>
```

Performance on the *eval* set can be obtained by running 
```
python eval_gise_mixtures.py --lbl_map meta/lbl_map.json --lmdb_path mixtures_lmdb/eval.lmdb --exp_dir <EXPERIMENT_DIR>/r<RUN_INDEX> --export_weights --export_dir exported_weights/r<RUN_INDEX> --results_csv <EXPERIMENT_DIR>/r<RUN_INDEX>/results.csv
```
The `--export_weights` option exports model `state_dict` as simple `.pth` files into `export_dir` for later use in transfer learning experiments. Performance metrics are written into `results.csv`, which can then averaged across runs.

Pre-trained models are provided can be found in `pretrained-models/experiments_60k_mixtures`. As previously mentioned, we only provide checkpoints for the best performing run for a CNN architecture, as opposed to the paper, which reports *average* results across runs. The following table depicts performance of each model; the first column lists the eval mAP across multiple runs, and `ckpt eval mAP` lists performance of the provided checkpoint.

| Model |  eval mAP (3-run avg) | ckpt eval mAP  | 
| ----- | ----- | ----- |
|       |       |       |
| ResNet-44| 0.5656 | 0.5689|
| ResNet-54| 0.5634 | 0.5671|
| ResNet-18| 0.5551 | 0.5635|
| ResNet-34| 0.5722 | 0.5783|
| ResNet-50| 0.5677 | 0.5727|
|          |      |     |
| EfficientNet-B0 | 0.5984 | 0.6032|
| EfficientNet-B1 | 0.6062 | 0.6097|
|          |      |     |
| DenseNet-121 | 0.6053 | 0.6136|


---
#### Transfer Learning Experiments
1. AudioSet

    The AudioSet video ids used for experiments are available in `data/audioset/`. Assuming you have downloaded the dataset and prepared `balanced_train.lmdb` and `eval.lmdb` for training and evaluation data, respectively, along with `lbl_map.json` that maps your string labels to integer indices, running AudioSet experiments is straightforward. No hyperparameter tuning is done on the eval set, and is used simply for EarlyStopping.

    1. Make sure you have exported `.pth` weights from ResNet-18 and EfficientNet-B1 architecures into appropriate filenames. Exported `.pth` weights for transfer learning are provided for transfer learning experiments. `pretrained-models/experiments_audioset`

    2. `experiments_audioset.sh` and `experiments_audioset_v100.sh` were used for ResNet-18 and EfficientNet-B1 experiments, respectively. 

    3. Once training is done, performance metrics can be calculated using the following script
    ```
    python eval_gise_mixtures.py --lbl_map data/audioset/lbl_map.json --lmdb_path <path to eval lmdb> --exp_dir <EXPERIMENT_DIR>_[ft,scratch]/r<RUN_INDEX> --results_csv <EXPERIMENT_DIR>_[ft,scratch]/r<RUN_INDEX>/results.csv --num_timesteps 1001
    ```
    You'll notice an additional parameter, `num_timesteps`. This parameter signifies number of timesteps center cropped from the spectrogram; 1001 corresponds to a 10 sec crop, which is the size of audioset clips.

    The following table lists performance of the provided checkpoint files.
    | Model | GISE-51-Mixtures Pretraining | eval mAP (3-run avg) | ckpt eval mAP  |
    | ----- | ----- | ----- | ----- |
    |       |       |       |       |
    | ResNet-18| False | 0.2053 | 0.2095|
    | EfficientNet-B1 | False | 0.2287 | 0.2317|
    | ResNet-18| True | 0.2236 | 0.2244|
    | EfficientNet-B1 | True | 0.2595 | 0.2603|

    <b>ATTENTION</b> Make sure your delimiter for string labels in your lmdb files is `;`, otherwise change the value for `delimiter` in audioset cfgs to desired value.
    

2. VGGSound

    As opposed to the previous experiments, which are multilabel multiclass tasks, VGGSound is multiclass classification task and is trained using the `train_classifier.py` script. Assuming you have obtained the dataset and prepared `train.lmdb`, `test.lmdb` as well as the corresponding `lbl_map.json`, view `experiments_vggsound_v100.sh` for more details. No hyperparameter tuning is done on the test set, and is used simply for EarlyStopping. VGGSound experiments were ran just once, and the provided checkpoints `pretrained-models/experiments_vggsound` have the same performance as mentioned in the paper.

    Evaluation of performance metrics for VGGSound can be done using `eval_vggsound.py`.
    ```
    python eval_vggsound.py --ckpt_path <path to ckpt> --lbl_map ./data/vggsound/lbl_map.json --lmdb_path <path to test.lmdb>
    ```

3. ESC-50

    Finally, to run ESC-50 experiments, download the provided ESC-50 lmdbs [here](https://gla-my.sharepoint.com/:f:/g/personal/2552300y_student_gla_ac_uk/EpZkbOWaqvtHkoJJGFUq774BlDcQwj5s-nMhf3vBVjvBOQ?e=PotHV9) and run `esc50_experiments.sh`. No need for further evaluation is needed since we're only concerned with fold-wise accuracy; just average best validation accuracy from train time `metrics.csv` results across runs across folds.
    
    <b>Attention:</b> For ESC-50, we provide checkpoints corresponding to best run for each fold. Thus, the effective 5-fold performance of models will be higher than that listed in the paper. More details are listed in the table below. More information can be found in `pretrained-models/experiments_esc50` folder.

    | Model |  Accuracy % (3-run avg) | Accuracy % of provided checkpoints |
    | ----- | -------- | ---------- |
    |       |          |            |
    | ResNet-18 | 83.92 | 85.35 |
    | EfficientNet-B1 | 85.72 | 86.75 |


## References
[1] Fonseca, Eduardo, et al. "FSD50k: an open dataset of human-labeled sound events." arXiv preprint arXiv:2010.00475 (2020).

[2] Gemmeke, Jort F., et al. "Audio set: An ontology and human-labeled dataset for audio events." 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.

[3] Chen, Honglie, et al. "Vggsound: A large-scale audio-visual dataset." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.

[4] Piczak, Karol J. "ESC: Dataset for environmental sound classification." Proceedings of the 23rd ACM international conference on Multimedia. 2015.

[5] Yadav, Sarthak et al. "GISE-51: A scalable isolated sound events dataset", arXiv preprint arXiv:2103.12306 (2021).
