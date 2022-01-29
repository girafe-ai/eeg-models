from eeg_models.eegnet import EegNet
from eeg_models.utils import n_parameters


def test_number_of_parameters():
    assert n_parameters(EegNet(n_classes=2)) == 2256
