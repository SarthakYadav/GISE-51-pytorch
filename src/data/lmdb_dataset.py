import os
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


# def load_bf_files_from_lmdb(bgfiles_lmdb):
#     env = lmdb.open(bgfiles_lmdb, subdir=os.path.isdir(bgfiles_lmdb),
#                     max_readers=1, readonly=True, lock=False,
#                     readahead=False, meminit=False)
#     with env.begin(write=False) as txn:
#         # self.length = txn.stat()['entries'] - 1
#         length = pa.deserialize(txn.get(b'__len__'))
#         keys = pa.deserialize(txn.get(b'__keys__'))
#     features = []
#     with env.begin(write=False) as txn:
#         for ix in tqdm.tqdm(range(length)):
#             byteflow = txn.get(keys[ix])
#             unpacked = pa.deserialize(byteflow)
#             features.append(unpacked)
#     env.close()
#     return features


class LMDBSpectrogramDataset(Dataset):
    def __init__(self, lmdb_path, labels_map,
                 audio_config, mode="multilabel",
                 augment=False, delimiter=",",
                 mixer=None, transform=None):
        super(LMDBSpectrogramDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = delimiter
        self.parse_audio_config(audio_config)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
        else:
            self.bg_files = None
        self.mode = mode
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
        self.env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = msgpack.unpackb(txn.get(b'__len__'))
            self.keys = msgpack.unpackb(txn.get(b'__keys__'))

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
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.unpackb(byteflow, object_hook=msgnp.decode)
        flac_audio = unpacked[0]
        lbls = unpacked[1]
        with io.BytesIO(flac_audio) as buf:
            preprocessed_audio, sr = sf.read(buf)
        label_tensor = self.__parse_labels__(lbls)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        return real, comp, label_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp, label_tensor = self.__get_item_helper__(index)
        if self.mixer is not None:
            real, final_label = self.mixer(self, real, label_tensor)
            if self.mode != "multiclass":

                return real, final_label
        return real, label_tensor

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            return self.labels_map[lbls]

    def __len__(self):
        return self.length

    def get_bg_len(self):
        return len(self.bg_features)
