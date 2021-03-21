import os
import math
import time
import io
import lmdb
import tqdm
import glob
import numpy as np
import librosa
import torch
import json
import random
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional
from src.data.audio_parser import AudioParser
import soundfile as sf
from src.data.utils import load_audio
import msgpack
import msgpack_numpy as msgnp


class InMemoryESC50Dataset(Dataset):
    def __init__(self, lmdb_path,
                 audio_config, fold_index=0,
                 augment=False,
                 mixer=None, transform=None, is_val=False):
        super(InMemoryESC50Dataset, self).__init__()
        assert audio_config is not None
        self.parse_audio_config(audio_config)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
        else:
            self.bg_files = None
        self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                       hop_length=self.hop_len, feature=self.feature_type)
        self.mixer = mixer
        self.transform = transform
        if self.bg_files is not None:
            print("prepping bg_features")
            self.bg_features = []
            for f in tqdm.tqdm(self.bg_files):
                preprocessed_audio = self.__get_audio__(f)
                real, comp = self.__get_feature__(preprocessed_audio)
                self.bg_features.append(real)
        else:
            self.bg_features = None

        self.buffered_audios = []
        self.label_strings = []
        self.fold_indices = []

        env_cp = os.environ.copy()
        self.num_local_gpus = len(env_cp.get('PL_TRAINER_GPUS', "0").split(","))
        self.local_rank = int(env_cp.get('LOCAL_RANK', "0"))
        print("self.local_rank: {} | self.num_local_gpus: {}".format(self.local_rank, self.num_local_gpus))
        self.is_val = is_val
        self.fold_index = fold_index
        self.prefetch_buffers(lmdb_path)
        self.length = len(self.buffered_audios)
        self.mode = "multiclass"

    def prefetch_buffers(self, lmdb_path):
        print("Prefetching buffers from lmdb")
        env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            length = msgpack.unpackb(txn.get(b'__len__'))
            keys = msgpack.unpackb(txn.get(b'__keys__'))
        num_samples = 0
        if self.is_val:
            # DDP calculates validation metrics for every process, and in the progress bar only reports
            # those from LOCAL_RANK=0 process.
            # since we're already not using DistributedSampler, to circumvent this behaviour
            # the easiest way is to load the full validation set for every process
            num_samples = length
            current_index_range = (0, num_samples)
        else:
            # in DDP each gpu sees a different portion of dataset (through DistributedSampler)
            # This loaded the entire training set as many times as there are GPUs
            # To counter this, sharded loading is implemented to load a 1/N_GPUS dataset in each DDP process
            # with no DistributedSampler

            # num_samples is padded to make equal parts on all GPUs: repetition!

            num_samples = int(math.ceil(length * 1.0 / self.num_local_gpus))
            total_size = num_samples * self.num_local_gpus
            indices_by_parts = [(ix, ix + num_samples) for ix in range(0, length, num_samples)]
            current_index_range = indices_by_parts[self.local_rank]
        print("[in prefetch_buffers] current_index_range: {} | node_rank: {}".format(current_index_range,
                                                                                     self.local_rank))
        folds = []
        audios = []
        labels = []
        with env.begin(write=False) as txn:
            for idx in range(current_index_range[0], current_index_range[1]):
                t0 = time.time()
                byteflow = txn.get(keys[idx])
                t1 = time.time()
                unpacked = msgpack.unpackb(byteflow, object_hook=msgnp.decode)
                flac_audio = unpacked[0]
                lbls = unpacked[1]
                fold_id = unpacked[2]
                audios.append(flac_audio)
                labels.append(lbls)
                folds.append(fold_id)
                if idx % 1000 == 0:
                    # time_per_1000 = time.time() - t0
                    print("DONE: {:07d}/{:07d} | Per Txn time:{}".format(idx, length, (t1 - t0) / 60))
                    # t0 = time_per_1000
        folds = np.asarray(folds)
        # print(folds)
        if self.is_val:
            idxs = np.where(folds == self.fold_index)[0]
        else:
            idxs = np.where(folds != self.fold_index)[0]
        idxs = idxs[np.random.permutation(len(idxs))]
        print(idxs)
        for idx in idxs:
            self.buffered_audios.append(audios[idx])
            self.label_strings.append(labels[idx])

    def parse_audio_config(self, audio_config):
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len
        self.normalize = audio_config.get("normalize", True)
        self.min_duration = audio_config.get("min_duration", None)
        self.background_noise_path = audio_config.get("bg_files", None)
        self.feature_type = audio_config.get("feature", "spectrogram")

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def get_bg_feature(self, index: int) -> torch.Tensor:
        if self.bg_features is None:
            return None
        real = self.bg_features[index]
        if self.transform is not None:
            real = self.transform(real)
        return real

    def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lbls = self.label_strings[index]
        flac_audio = self.buffered_audios[index]
        with io.BytesIO(flac_audio) as buf:
            preprocessed_audio, sr = sf.read(buf)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        return real, comp, torch.tensor(lbls)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp, label_tensor = self.__get_item_helper__(index)
        if self.mixer is not None:
            real, final_label = self.mixer(self, real, label_tensor)
            if self.mode != "multiclass":
                return real, final_label
        return real, label_tensor

    # def __parse_labels__(self, lbls: str) -> torch.Tensor:
    #     if self.mode == "multilabel":
    #         label_tensor = torch.zeros(len(self.labels_map)).float()
    #         for lbl in lbls.split(self.labels_delim):
    #             label_tensor[self.labels_map[lbl]] = 1
    #
    #         return label_tensor
    #     elif self.mode == "multiclass":
    #         return self.labels_map[lbls]

    def __len__(self):
        return self.length

    def get_bg_len(self):
        return len(self.bg_features)
