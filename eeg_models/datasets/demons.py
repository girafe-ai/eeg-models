import numpy as np
from somepytools.typing import Callable, Dict, Directory, Optional, Sequence, Tuple

from .. import data_dir
from .abstract import AbstractEegDataset
from .constants import SPLIT_TRAIN


class DemonsP300Dataset(AbstractEegDataset):
    # from somepytools.constants import TIME_CONSTANTS
    _ms_in_sec = 1000
    _hdf_path = "p300dataset"
    _ds_folder_name = "demons"

    _act_dtype = np.dtype(
        [
            ("id", np.int),
            ("target", np.int),
            ("is_train", np.bool),
            ("prediction", np.int),
            ("sessions", np.object),  # list of `_session_dtype`
        ]
    )
    _session_dtype = np.dtype(
        [
            ("eeg", np.object),
            ("starts", np.object),
            ("stimuli", np.object),
        ]
    )

    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = SPLIT_TRAIN,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        if root is None:
            root = data_dir / "demons"
        super().__init__(root, split, transforms, transform, target_transform, download)

    @property
    def channels(self) -> Tuple[str]:
        return ("Cz", "P3", "Pz", "P4", "PO3", "PO4", "O1", "O2")

    @property
    def sampling_rate(self) -> float:
        return 500.0

    @property
    def urls(self) -> Dict[str, Sequence[str]]:
        return {
            "origin": [
                "https://gin.g-node.org/v-goncharenko/neiry-demons/raw/master/nery_demons_dataset.zip"
            ]
        }

    def download(self):
        pass  # nothing to do - this is DVC based dataset
