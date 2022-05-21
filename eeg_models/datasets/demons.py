import h5py
import numpy as np
import pandas as pd
import torch
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
        sample_per_epoch: int = None,
    ):
        if root is None:
            root = data_dir / self._default_root
        super().__init__(root, split, transforms, transform, target_transform, download)

        self.meta = pd.read_csv(self.root / "meta.csv")

        if sample_per_epoch is None:
            self.sample_per_epoch = 400
        else:
            self.sample_per_epoch = sample_per_epoch

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        record = self._read_hdf(self.root / self.meta.loc[index, "filename"])

        runs_raw = {}
        for i, act in enumerate(record):
            # target and stims are increased by 1
            # because the channel is filled with zeros by default
            target = act["target"] + 1
            run_data = []
            for eeg, starts, stims in act["sessions"]:
                starts = starts * self.sampling_rate / self._ms_in_sec
                starts = starts.round().astype(np.int)
                stims = stims + 1
                stims_channel = np.zeros(eeg.shape[1])
                target_channel = np.zeros(eeg.shape[1])

                for start, stimul in zip(starts, stims):
                    stims_channel[start] = stimul
                    target_channel[start] = 1 if stimul == target else 2

                round_data = np.vstack(
                    (eeg, stims_channel[None, :], target_channel[None, :])
                )
                run_data.append(round_data)

            raw = np.hstack(run_data)
            runs_raw[f"run_{i}"] = raw
        data = {"session_0": runs_raw}
        NROWS = 170
        nrows = NROWS

        data_tab, label_tab, index_tab = self._get_data_tab(data, nrows)

        # Transforms :
        filtered_eeg = []
        filtered_label = []
        filtered_index = []
        for i in range(NROWS):
            raw_eegs = data_tab[i]
            self.transform.fit(raw_eegs)
        for i in range(NROWS):
            raw_eegs = data_tab[i]
            result = np.stack(self.transform.transform(raw_eegs))
            filtered_eeg.append(result)
            filtered_label.append(label_tab[i])
            filtered_index.append(index_tab[i])
        output = (
            torch.as_tensor(np.stack(filtered_eeg)).float(),
            torch.as_tensor(np.stack(filtered_label)).float(),
        )
        # shuffling
        indices = torch.randperm(output[0].size()[0])
        o1 = output[0][indices]
        o2 = output[1][indices]
        output = (o1, o2)
        return output

    def _get_data_tab(self, data, nrows):

        # self.sample_per_epoch  :   number of samples before decimation
        # nrows : number of sub-epochs extrated from each subject file

        data_tab = []
        label_tab = []
        index_tab = []
        NROWS = nrows

        # for each session in subject file
        for i in range(len(data)):
            #  for each run of a session
            for j in range(len(data[f"session_{i}"])):
                data_tmp = []
                label_tmp = []
                index_tmp = []
                i_low = 200
                i_high = i_low + self.sample_per_epoch
                m_low = -100
                m_high = m_low + self.sample_per_epoch
                # find event of interest
                signal = data[f"session_{i}"][f"run_{j}"][9]
                signal_test = [
                    (indice, signal[indice])
                    for indice in range(len(signal))
                    if signal[indice] == 1
                ]
                # for each event create one epoch of self.sample_per_batch samples
                if signal_test == []:
                    for k in range(10):
                        # limits = ( 300 + k * 1400, 700 + k * 1400)
                        limits = (i_low + k * 1400, i_high + k * 1400)
                        if (
                            data[f"session_{i}"][f"run_{j}"][
                                0:8, limits[0] : limits[1]
                            ].shape[1]
                            == self.sample_per_epoch
                        ):
                            data_tmp.append(
                                data[f"session_{i}"][f"run_{j}"][
                                    0:8, limits[0] : limits[1]
                                ]
                            )
                            label_tmp.append(np.array([0.0]))
                            index_tmp.append(np.array([limits[0], limits[1]]))
                if signal_test != []:
                    for k in range(len(signal_test)):
                        current_indice = signal_test[k][0]
                        if (
                            data[f"session_{i}"][f"run_{j}"][
                                0:8, current_indice + m_low : current_indice + m_high
                            ].shape[1]
                            == self.sample_per_epoch
                        ):
                            data_tmp.append(
                                data[f"session_{i}"][f"run_{j}"][
                                    0:8, current_indice + m_low : current_indice + m_high
                                ]
                            )
                            label_tmp.append(np.array([1.0]))
                            index_tmp.append(
                                np.array(
                                    [current_indice + m_low, current_indice + m_high]
                                )
                            )
                data_tab += data_tmp
                label_tab += label_tmp
                index_tab += index_tmp
            data_tab = (np.stack(data_tab))[:NROWS]
            label_tab = (np.stack(label_tab))[:NROWS]
            index_tab = (np.stack(index_tab))[:NROWS]
        return data_tab, label_tab, index_tab

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
