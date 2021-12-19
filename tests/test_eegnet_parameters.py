from eeg_models.EegNet import EegNet


def test_number_of_parameters():
    assert sum(p.numel() for p in EegNet(n_classes=2).parameters()) == 2256
