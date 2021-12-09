import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Pytorch Implementation of EEGNet

    While the original paper used Dropout, we found that Dropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, Dropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
        advised to do some model searching to get optimal performance on your
        particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Dropout2D or Dropout, passed as a string.
    """

    def __init__(
        self,
        nb_classes,
        Chans=64,
        Samples=128,
        dropoutRate=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        dropoutType="Dropout",
    ):
        super(EEGNet, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutType = dropoutType

        if dropoutType == "Dropout2D":
            dropoutType = nn.Dropout2d(p=dropoutRate)
        elif self.dropoutType == "Dropout":
            dropoutType = nn.Dropout(p=dropoutRate)
        else:
            raise ValueError(
                "dropoutType must be one of Dropout2D " "or Dropout, passed as a string."
            )

        # Block 1
        self.padding_1 = nn.ZeroPad2d((kernLength // 2 - 1, kernLength // 2, 0, 0))
        self.conv_1 = nn.Conv2d(1, F1, (1, kernLength), bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(F1, False)
        self.depthwise_1 = nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(D * F1, False)
        self.avgpool_1 = nn.AvgPool2d(1, 4)
        self.dropout_1 = dropoutType

        # Block 2
        self.padding_2 = nn.ZeroPad2d((kernLength // 8 - 1, kernLength // 8, 0, 0))
        self.seperate_1 = nn.Conv2d(
            D * F1, D * F1, (1, kernLength // 4), groups=F1 * D, bias=False
        )
        self.seperate_2 = nn.Conv2d(D * F1, D * F1, 1, bias=False)
        self.batchnorm_3 = nn.BatchNorm2d(D * F1, False)
        self.avgpool_2 = nn.AvgPool2d(1, 8)
        self.dropout_2 = dropoutType

        # FC Layer
        self.fc1 = nn.Linear(F2 * (Samples // 32), nb_classes, bias=False)

    def forward(self, input1):

        input1 = input1.view(1, 1, self.Chans, self.Samples)
        # Block 1
        block1 = self.padding_1(input1)
        block1 = self.conv_1(block1)
        block1 = self.batchnorm_1(block1)
        block1 = self.depthwise_1(block1)
        block1 = self.batchnorm_2(block1)
        block1 = F.elu(block1)
        block1 = self.avgpool_1(block1)
        block1 = self.dropout_1(block1)
        # Block 2
        block2 = self.padding_2(block1)
        block2 = self.seperate_1(block2)
        block2 = self.seperate_2(block2)
        block2 = self.batchnorm_3(block2)
        block2 = F.elu(block2)
        block2 = self.avgpool_2(block2)
        block2 = self.dropout_2(block2)
        # FC Layer
        flatten = block2.view(-1, self.F2 * (self.Samples // 32))
        dense = self.fc1(flatten)
        softmax = F.softmax(dense, dim=1)

        return softmax
