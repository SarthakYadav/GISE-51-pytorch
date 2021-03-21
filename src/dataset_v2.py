import os
import torch
import json
import random
import librosa
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from src.data.spectrogram import Spectrogram
from typing import Tuple, Optional


def load_audio_data(x, sr=22050, end_time=None):
    # if end_time is None:
    #     # belongs to site id 3, load whole audio
    #     x, sr = librosa.load(f, sr=sr, mono=True)
    # else:
    #     # other sites are recorded this way
    #     offset = end_time - 5
    #     x, sr = librosa.load(f, sr=sr, mono=True, offset=offset, duration=5)
    #
    # if len(x) == 0:
    #     raise ValueError("length of array cannot be zero.")

    pre_emphasis = 0.97
    x = np.append(x[0], x[1:] - pre_emphasis * x[:-1])
    normalizedy = librosa.util.normalize(x)
    D = librosa.stft(normalizedy, n_fft=511, hop_length=380)
    spec, _ = librosa.magphase(D)
    sp = 20 * np.log10(spec+1e-9)

    return torch.tensor(sp, dtype=torch.float32), torch.tensor(D, dtype=torch.complex64)


class SpectrogramDataset(Dataset):
    def __init__(self, manifest_path: str, labels_map: str,
                 audio_config: dict, augment: Optional[bool] = False,
                 labels_delimiter: Optional[str] = ",") -> None:
        super(SpectrogramDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)

        self.len = None
        self.labels_delim = labels_delimiter
        df = pd.read_csv(manifest_path)
        self.files = df['files'].values
        self.labels = df['labels'].values
        assert len(self.files) == len(self.labels)
        self.len = len(self.files)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = int(self.sr * win_len)
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = int(self.sr * hop_len)
        self.hop_len = 380
        self.normalize = audio_config.get("normalize", True)
        self.augment = augment
        self.clip_duration = audio_config.get("clip_duration", None)
        if self.clip_duration:
            self.clip_duration = int(self.sr * self.clip_duration)
        self.spec_parser = Spectrogram(n_fft=self.n_fft, win_length=self.win_len,
                                       hop_length=self.hop_len, normalized=self.normalize)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        lbls = self.labels[index]
        label_tensor = self.__parse_labels__(lbls)
        audio, sr = torchaudio.load(f)
        # print(audio.shape)
        assert sr == self.sr
        if self.augment:
            if self.clip_duration is not None:
                if not self.clip_duration > audio.shape[1]:
                    start_idx = random.randint(0, audio.shape[1] - 1)
                    if start_idx + self.clip_duration > audio.shape[1]:
                        offset = (start_idx + self.clip_duration) - audio.shape[1]
                        start_idx -= offset
                else:
                    start_idx = 0
                audio = audio[:, start_idx:start_idx + self.clip_duration]
        real, comp = load_audio_data(audio.squeeze().cpu().numpy())
        # real, comp = self.spec_parser(audio)
        # real = real[0]  # torchaudio has a batch axis
        # comp = comp[0]
        return real, comp, label_tensor

    def __parse_labels__(self, lbls: list) -> torch.Tensor:
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(self.labels_delim):
            label_tensor[self.labels_map[lbl]] = 1
        return label_tensor

    def __len__(self):
        return self.len


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        complex_tensor = sample[1]
        target = sample[2]
        seq_length = real_tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
        inputs_complex[x][0].narrow(1, 0, seq_length).copy_(complex_tensor)
        targets.append(target.unsqueeze(0))
    targets = torch.cat(targets)
    return inputs, inputs_complex, targets
