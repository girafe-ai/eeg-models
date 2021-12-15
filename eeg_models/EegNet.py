import torch.nn as nn


class EegNet(nn.Sequential):
    def __init__(
        self, n_classes, chans=64, samples=128, dropoutRate=0.5, rate=64, f1=8, d=2, f2=16
    ):
        super().__init__(
            nn.ZeroPad2d((rate // 2 - 1, rate // 2, 0, 0)),
            nn.Conv2d(1, f1, (1, rate), bias=False),
            nn.BatchNorm2d(f1, False),
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1, False),
            nn.ELU(),
            nn.AvgPool2d(1, 4),
            nn.Dropout(p=dropoutRate),
            nn.ZeroPad2d((rate // 8 - 1, rate // 8, 0, 0)),
            nn.Conv2d(d * f1, d * f1, (1, rate // 4), groups=f1 * d, bias=False),
            nn.Conv2d(d * f1, d * f1, 1, bias=False),
            nn.BatchNorm2d(d * f1, False),
            nn.ELU(),
            nn.AvgPool2d(1, 8),
            nn.Dropout(p=dropoutRate),
            nn.Flatten(),
            nn.Linear(f2 * (samples // 32), n_classes, bias=False),
            nn.Softmax(),
        )
