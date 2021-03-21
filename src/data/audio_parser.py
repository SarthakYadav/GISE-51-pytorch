import os
import torch
import torchaudio
import glob
import soundfile as sf
import numpy as np
import librosa
from src.data.functional_utils import spectrogram


class AudioParser(object):
    def __init__(self, n_fft=511, win_length=None, hop_length=None, sample_rate=22050,
                 feature="spectrogram", top_db=150, return_mean=False):
        super(AudioParser, self).__init__()
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.n_fft // 2 if hop_length is None else hop_length
        assert feature in ['melspectrogram', 'spectrogram', "torchspectrogram"]
        self.feature = feature
        self.top_db = top_db
        if feature == "melspectrogram":
            self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=96 * 20,
                                                                win_length=int(sample_rate * 0.03),
                                                                hop_length=int(sample_rate * 0.01),
                                                                n_mels=96)
        else:
            self.melspec = None

        if feature == "torchspectrogram":
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB("magnitude", top_db=self.top_db)
            self.window = torch.hann_window(self.win_length)
        else:
            self.amp_to_db = None

        self.return_mean = return_mean

    def __call__(self, audio):
        if self.feature == 'spectrogram':
            # TOP_DB = 150
            comp = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length)
            real = np.abs(comp)
            real = librosa.amplitude_to_db(real, top_db=self.top_db)
            real += self.top_db / 2
            mean = real.mean()
            real -= mean
            if self.return_mean:
                return real, comp, mean
            else:
                return real, comp

        elif self.feature == 'torchspectrogram':
            # TOP_DB = 150
            x = torch.from_numpy(audio).unsqueeze(0).float()
            spec = spectrogram(x, 0, self.window, self.n_fft, self.hop_length, self.win_length, None)
            spec_real = torch.abs(spec)
            spec_real = self.amp_to_db(spec_real)
            spec_real += self.top_db / 2
            mean = spec_real.mean()
            spec_real -= mean
            spec_real = spec_real[0].numpy().astype('float32')
            if self.return_mean:
                return spec_real, spec, mean
            else:
                return spec_real, spec

        elif self.feature == 'melspectrogram':
            # these are params from FSDKaggle2019 winner
            # specgram = librosa.feature.melspectrogram(audio, sr=22050,
            #                                           n_fft=128*20, hop_length=345*2,
            #                                           n_mels=128)

            x = torch.from_numpy(audio).unsqueeze(0).float()
            specgram = self.melspec(x)[0].numpy()
            specgram = librosa.power_to_db(specgram)
            specgram = specgram.astype('float32')
            specgram += self.top_db / 2
            mean = specgram.mean()
            specgram -= mean
            if self.return_mean:
                return specgram, specgram, mean
            else:
                return specgram, specgram
