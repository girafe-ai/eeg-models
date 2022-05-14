import h5py
import numpy as np
import pandas as pd
from somepytools.typing import (
    Any,
    Array,
    Callable,
    Dict,
    Directory,
    File,
    Optional,
    Sequence,
    Tuple,
)

from .. import data_dir
from .abstract import AbstractEegDataset
from .constants import SPLIT_TRAIN


class DemonsP300Dataset(AbstractEegDataset):
    _default_root = "demons_p300_dataset"

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
            root = data_dir / self._default_root
        super().__init__(root, split, transforms, transform, target_transform, download)

        self.meta = pd.read_csv(self.root / "meta.csv")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._read_hdf(self.root / self.meta.loc[index, "filename"])

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

    @classmethod
    def _read_hdf(cls, filename: File) -> Array:
        """Reads data from HDF file

        Returns:
            array of `_act_dtype`
        """
        with h5py.File(filename.as_posix(), "r") as hfile:
            group = hfile[cls._hdf_path]
            record = np.empty(len(group), cls._act_dtype)
            for i, act in enumerate(group.values()):
                record[i]["sessions"] = np.array(
                    [cls._strip(item) for item in act], cls._session_dtype
                )
                for name, value in act.attrs.items():
                    record[i][name] = value
        return record

    @staticmethod
    def _strip(session) -> tuple:
        """Strips nans (from right side of all channels) added during hdf5 packaging

        Returns:
            tuple ready to be converted to `_session_dtype`
        """
        eeg, *rest = session
        ind = -next(i for i, value in enumerate(eeg[0, ::-1]) if not np.isnan(value))
        if ind == 0:
            ind = None
        return tuple((eeg[:, :ind], *rest))
