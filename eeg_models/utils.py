import os
import os.path as osp

from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose
from pooch import file_hash, retrieve

from eeg_models.types import Any, Directory, Optional


def get_dataset_path(sign: str, path: Optional[Directory]) -> Optional[Directory]:
    sign = sign.upper()
    key = "MNE_DATASETS_{:s}_PATH".format(sign)
    if get_config(key) is None:
        if get_config("MNE_DATA") is None:
            path_def = osp.join(osp.expanduser("~"), "mne_data")
            print(
                "MNE_DATA is not already configured. It will be set to "
                "default location in the home directory - "
                + path_def
                + "\nAll datasets will be downloaded to this location, if anything is "
                "already downloaded, please move manually to this location"
            )
            if not osp.isdir(path_def):
                os.makedirs(path_def)
            set_config("MNE_DATA", osp.join(osp.expanduser("~"), "mne_data"))
        set_config(key, get_config("MNE_DATA"))
    return _get_path(path, key, sign)


@verbose
def data_path(
    url: str,
    sign: str,
    path: Optional[Directory] = None,
    force_update: bool = False,
    update_path: bool = True,
    verbose: bool = None,
) -> Optional[Directory]:
    path = get_dataset_path(sign, path)
    key_dest = "MNE-{:s}-data".format(sign.lower())
    destination = _url_to_local_path(url, osp.join(path, key_dest))
    if not osp.isfile(destination) or force_update:
        if osp.isfile(destination):
            os.remove(destination)
        if not osp.isdir(osp.dirname(destination)):
            os.makedirs(osp.dirname(destination))
        retrieve(url, None, path=destination)
    return destination


@verbose
def data_dl(
    url: str,
    sign: str,
    path: Optional[Directory] = None,
    force_update: bool = False,
    verbose: bool = None,
) -> Optional[Directory]:
    path = get_dataset_path(sign, path)
    key_dest = "MNE-{:s}-data".format(sign.lower())
    destination = _url_to_local_path(url, osp.join(path, key_dest))

    if not osp.isfile(destination) or force_update:
        if osp.isfile(destination):
            os.remove(destination)
        if not osp.isdir(osp.dirname(destination)):
            os.makedirs(osp.dirname(destination))
        known_hash = None
    else:
        known_hash = file_hash(destination)
    dlpath = retrieve(
        url, known_hash, fname=osp.basename(url), path=osp.dirname(destination)
    )
    return dlpath


def n_parameters(model: Any) -> int:
    """Calculates number of parameters in the model.

    Args:
        model: Model to count parameters in.

    Returns: Number of parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())
