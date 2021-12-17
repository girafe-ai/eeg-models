from eeg_models.EegNet import EegNet


net = EegNet(n_classes=2)


def test_number_of_parameters():
    assert sum(p.numel() for p in net.parameters()) == 2256
