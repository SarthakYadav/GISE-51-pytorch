# GISE-51-pytorch

Official code release for the paper [GISE-51: A scalable isolated sound events dataset]() in pytorch using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning). Instructions and implementation for replicating all baseline experiments, including pre-trained models. 

## About
Most  of  the  existing  isolated  sound  event  datasets  comprisea small number of sound event classes,  usually 10 to 15,  re-stricted to a small domain, such as domestic and urban soundevents. In this work, we introduce GISE-51, a dataset spanning51 isolated sound event classes belonging to a broad domain ofevent types. We also release GISE-51-Mixtures, a dataset of 5-second soundscapes with hard-labelled event boundaries synthe-sized from GISE-51 isolated sound events. We conduct baselinesound event recognition (SER) experiments on the GISE-51-Mixtures dataset, benchmarking prominent convolutional neu-ral networks, and models trained with the dataset demonstratestrong transfer learning performance on existing audio recog-nition benchmarks. Together, GISE-51 and GISE-51-Mixtures attempt to address some of the shortcomings of recent soundevent datasets, providing an open, reproducible benchmark forfuture research along with the freedom to adapt the includedisolated  sound  events  for  domain-specific  applications.

This repository contains code for reproducing all experiments done in the paper. For more information on the dataset, as well as the pre-trained models, visit [GISE-51: A scalable isolated sound events dataset]()

## Keypoints
### In-memory data loading
* The implementation utilizes `lmdb` format for dataset serialization, and loads data in-memory while training (`src/data/inmemory_lmdb_dataset.py`). To keep memory usage to a minimum, the records in the lmdb's are not numpy arrays; they're entire flac file bytes written into lmdb's. For more info view `pack_mixtures_into_lmdb.py`.
* This was done because majority of the experiments were done on resource with slow I/O performance.  This had the added benefit of great GPU utilization, and works well for our dataset use cases. 
* One can easily modify `src/data/inmemory_lmdb_dataset.py` if you need a regular dataset that reads data from disk on-the-fly.

### Pre-trained models
As mentioned in the paper, we performed experiments several times and reported average results. However, since releasing so many models would be extremely cumbersome, only the best performing for an experiment is released. If you need all the best performing checkpoints from all the runs, please get in touch.

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
There are two options for generating lmdb files:
1. Downloading provided tar archives and pack them into lmdbs. <b>(Recommended)</b>
2. Downloading `noises.tar.gz`, `isolated_events.tar.gz` and `mixtures_jams.tar.gz`, generate soundscapes locally using scaper from jams files, convert soundscapes to `.flac` (for lower memory usage) using `scripts/wav2flac.py` and pack them into mixtures.

We recommend using method 1. For packing soundscape mixtures into lmdbs:

* Download tar archive for corresponding data subset from [dataset page]()
* Run `pack_mixtures_into_lmdb.py` to generate lmdb files from soundscape mixtures as follows. 
   ```
    # for generating lmdb for 60k soundscapes
    python pack_mixtures_into_lmdb.py --mixture_dir "mixtures_flac/train" --lmdb_path mixtures_lmdb/train.lmdb --map_size 2e10

    # for generating val lmdb
    python pack_mixtures_into_lmdb.py --mixture_dir "mixtures_flac/val" --lmdb_path mixtures_lmdb/val.lmdb --map_size 2e10

    # for generating eval lmdb
    python pack_mixtures_into_lmdb.py --mixture_dir "mixtures_flac/eval" --lmdb_path mixtures_lmdb/eval.lmdb --map_size 2e10
   ```
* To generate lmdbs for replicating Experiment 4.1 from the paper, view `generate_mixtures.sh`. lmdbs for AudioSet, VGGSound and ESC-50 experiments can be generated in a similar manner. ESC-50 lmdbs can be found [here]().

### Experiments
The following sections outline how to reproduce experiments conducted in the paper. Before running any experiment, make sure your paths are correct in the corresponding `.cfg` files.

---
#### Number of synthesized soundscapes v/s val mAP
This section studies how val mAP performance scales with number of training mixtures, using ResNet-18 architecture.

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
This will write evaluation metrics for <RUN_INDEX> in `results.csv`. One can then average results across runs. `lbl_map.json` contains a json serialized dictionary that maps dataset labels to corresponding integer indices.

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

Pre-trained models are provided [here](). As previously mentioned, we only provide checkpoints for the best performing run for a CNN architecture, as opposed to the paper, which reports *average* results across runs. The following table lists the checkpoints and their runs.

---
#### Transfer Learning Experiments
##### AudioSet

The AudioSet video ids used for experiments are available in `data/audioset/`. Assuming you have downloaded the dataset and prepared `balanced_train.lmdb` and `eval.lmdb` for training and evaluation data, respectively, along with `lbl_map.json` that maps your string labels to integer indices, running AudioSet experiments is straightforward. 

1. Make sure you have exported `.pth` weights from ResNet-18 and EfficientNet-B1 architecures into appropriate filenames. You can also directly download exported `.pth` weights for transfer learning [here]()

2. `experiments_audioset.sh` and `experiments_audioset_v100.sh` were used for ResNet-18 and EfficientNet-B1 experiments, respectively. 

3. Once training is done, performance metrics can be calculated using the following script
   ```
   python eval_gise_mixtures.py --lbl_map audioset/meta/lbl_map.json --lmdb_path audioset/lmdbs/eval.lmdb --exp_dir <EXPERIMENT_DIR>_[ft,scratch]/r<RUN_INDEX> --results_csv <EXPERIMENT_DIR>_[ft,scratch]/r<RUN_INDEX>/results.csv --num_timesteps 1001
   ```
You'll notice an additional parameter, `num_timesteps`. This parameter signifies number of timesteps center cropped from the spectrogram; 1001 corresponds to a 10 sec crop, which is the size of audioset clips.

<b>ATTENTION</b> Make sure your delimiter for string labels in your lmdb files is `;`, otherwise change the value for `delimiter` in audioset cfgs to desired value.

##### VGGSound


## References
[1] Fonseca, E., Favory, X., Pons, J., Font, F. and Serra, X., 2020. FSD50k: an open dataset of human-labeled sound events. arXiv preprint arXiv:2010.00475.  
