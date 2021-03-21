import torch
from torch import Tensor
import torchaudio
from typing import Optional, Callable, Tuple
from src.data.functional_utils import spectrogram


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 # power: Optional[float] = None,       # good for what we doin
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = None
        self.normalized = normalized
        print("NFFT: {} | WIN_LEN: {} | HOP_LEN: {}".format(self.n_fft, self.win_length, self.hop_length))

    def forward(self, waveform: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        spec = spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
                           self.win_length, self.power)

        # with our current settings, vis power == None, we get complex spectrogram
        spec_real = spec.abs()
        # spec_real = torch.log1p(spec_real)
        spec_real = torch.log1p(spec_real)
        # spec_real = 20 * torch.log10(spec_real+1e-9)
        return spec_real, spec
        # return F.spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
        #                      self.win_length, self.power, self.normalized)
