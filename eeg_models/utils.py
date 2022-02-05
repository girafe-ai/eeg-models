import os
from os import path as op

from mne.datasets.utils import _do_path_update, _get_path
from mne.utils import _fetch_file, _url_to_local_path, verbose

from eeg_models.types import Any


@verbose
def dt_path(url, sign, path=None, force_update=False, update_path=True, verbose=None):
    sign = sign.upper()
    key = "MNE_DATASETS_{:s}_PATH".format(sign)
    key_dest = "MNE-{:s}-data".format(sign.lower())
    path = _get_path(path, key, sign)
    destination = _url_to_local_path(url, op.join(path, key_dest))
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    _do_path_update(path, update_path, key, sign)
    return destination


def n_parameters(model: Any) -> int:
    """Calculates number of parameters in the model.

    Args:
        model: Model to count parameters in.

    Returns: Number of parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())
