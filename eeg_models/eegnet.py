import torch.nn as nn


class EegNet(nn.Sequential):
    """Pytorch Implementation of EegNet.

    This implementation is mostly based on the original paper "EEGNet: A Compact Convolutional Network
    for EEG-based Brain-Computer Interfaces" which can be found on the "https://arxiv.org/abs/1611.08024"
    and repository is: 'https://github.com/vlawhern/arl-eegmodels'.

    Assumes the input signal is sampled at 128Hz and used 64 channels. If you want to use this model
    for any other sampling rate and channels, you will need to modify the lengths of temporal
    kernels and average pooling size.

    Args:
      n_classes: number of classes to classify.
      n_channels: number of channels in the EEG data.
      n_samples: number of time points in the EEG data.
      dropout_rate: dropout fraction.
      rate: sampling rate in in Hertz (Hz).
      f1: number of temporal filters.
      d: number of spatial filters.
      f2: number of pointwise filters.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 64,
        n_samples: int = 128,
        dropout_rate: float = 0.5,
        rate: int = 128,
        f1: int = 8,
        d: int = 2,
        f2=None,
        device=None,
        dtype=None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        kernel = rate // 2
        if f2 is None:
            f2 = f1 * d
        super().__init__(
            nn.Unflatten(0, (1, 1, n_channels)),
            nn.Conv2d(1, f1, (1, kernel), padding="same", bias=False).to(
                device=self.device, dtype=self.dtype
            ),
            nn.BatchNorm2d(f1, track_running_stats=False).to(
                device=self.device, dtype=self.dtype
            ),
            nn.Conv2d(f1, d * f1, (n_channels, 1), groups=f1, bias=False).to(
                device=self.device, dtype=self.dtype
            ),
            nn.BatchNorm2d(d * f1, track_running_stats=False).to(
                device=self.device, dtype=self.dtype
            ),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                d * f1,
                d * f1,
                (1, kernel // 4),
                padding="same",
                groups=f1 * d,
                bias=False,
            ).to(device=self.device, dtype=self.dtype),
            nn.Conv2d(d * f1, d * f1, 1, bias=False).to(
                device=self.device, dtype=self.dtype
            ),
            nn.BatchNorm2d(d * f1, track_running_stats=False).to(
                device=self.device, dtype=self.dtype
            ),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(f2 * (n_samples // 32), n_classes, False).to(
                device=self.device, dtype=self.dtype
            ),
        )
