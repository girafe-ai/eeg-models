import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        input = input.view(self.shape)
        return input


class EegNet(nn.Sequential):
    """Pytorch Implementation of EegNet.

    This implementation is mostly based on the original paper which is models can be found on the
    'https://github.com/vlawhern/arl-eegmodels' that is implemented in Keras.

    Assumes the input signal is sampled at 128Hz and used 64 channels. If you want to use this model
    for any other sampling rate and channels, you will need to modify the lengths of temporal
    kernels and average pooling size.

    Args:
      n_classes: number of classes to classify.
      n_channels: number of channels in the EEG data.
      n_samples: number of time points in the EEG data.
      dropout_rate: dropout fraction.
      rate: sampling rate.
      kernel: length of temporal convolution in first layer. It is equal to half the sampling rate.
      f1: number of temporal filters default: f1 = 8.
      f2: number of pointwise filters default: f2 = f1 * d.
      d: number of spatial filters default: d = 2.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels=64,
        n_samples=128,
        dropout_rate=0.5,
        rate=128,
        f1=8,
        d=2,
        f2=None,
        device=None,
        dtype=None,
    ):
        kernel = rate // 2
        if f2 is None:
            f2 = f1 * d
        super().__init__(
            View((1, 1, n_channels, n_samples)),
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, kernel),
                padding="same",
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm2d(
                num_features=f1, track_running_stats=False, device=device, dtype=dtype
            ),
            nn.Conv2d(
                in_channels=f1,
                out_channels=d * f1,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm2d(
                num_features=d * f1, track_running_stats=False, device=device, dtype=dtype
            ),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(
                in_channels=d * f1,
                out_channels=d * f1,
                kernel_size=(1, kernel // 4),
                padding="same",
                groups=f1 * d,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.Conv2d(
                in_channels=d * f1,
                out_channels=d * f1,
                kernel_size=1,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm2d(
                num_features=d * f1, track_running_stats=False, device=device, dtype=dtype
            ),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate),
            nn.Flatten(),
            nn.Linear(
                in_features=f2 * (n_samples // 32),
                out_features=n_classes,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.Softmax(),
        )
