import collections
import json

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
        remove_inds_fname: Optional[File] = None,
        split: str = SPLIT_TRAIN,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        sample_per_epoch: int = None,
        use_cache=False,
    ):
        if root is None:
            root = data_dir / self._default_root
        super().__init__(root, split, transforms, transform, target_transform, download)

        if remove_inds_fname is not None:
            with open(remove_inds_fname) as f:
                self.remove_inds = json.load(f)
        else:
            self.remove_inds = None

        self.meta = pd.read_csv(self.root / "meta.csv")

        if sample_per_epoch is None:
            self.sample_per_epoch = 400
        else:
            self.sample_per_epoch = sample_per_epoch

        self.cached_data = collections.defaultdict(lambda: None)

        self.use_cache = use_cache

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        if self.use_cache and self.cached_data[index] is None:

            output = self._internal_get_data(index)
            return output

            # record = self._read_hdf(self.root / self.meta.loc[index, "filename"])

            # runs_raw = {}
            # for i, act in enumerate(record):
            #     # target and stims are increased by 1
            #     # because the channel is filled with zeros by default
            #     target = act["target"] + 1
            #     run_data = []
            #     for eeg, starts, stims in act["sessions"]:
            #         starts = starts * self.sampling_rate / self._ms_in_sec
            #         starts = starts.round().astype(np.int)
            #         stims = stims + 1
            #         stims_channel = np.zeros(eeg.shape[1])
            #         target_channel = np.zeros(eeg.shape[1])

            #         for start, stimul in zip(starts, stims):
            #             stims_channel[start] = stimul
            #             target_channel[start] = 1 if stimul == target else 2

            #         round_data = np.vstack(
            #             (eeg, stims_channel[None, :], target_channel[None, :])
            #         )
            #         run_data.append(round_data)

            #     raw = np.hstack(run_data)
            #     runs_raw[f"run_{i}"] = raw
            # data = {"session_0": runs_raw}

            # # NROWS : int, number of rows in the dataset output by the loader
            # # 250  = 5 events par period x  10 periods per run x 5 runs with events "1"
            # # runs with no "1" events are not included and used
            # NROWS = 250
            # nrows = NROWS

            # data_tab, label_tab, index_tab = self._get_data_tab(data, nrows)

            # # list of outliers
            # if (self.remove_inds is not None) and (str(index) in self.remove_inds.keys()):
            #     list_indice = list(self.remove_inds[str(index)])
            #     list_ind_to_remove = [
            #         int(list_indice[j]) for j in range(len(list_indice))
            #     ]
            # else:
            #     list_ind_to_remove = []

            # # Transforms :
            # filtered_eeg = []
            # filtered_label = []
            # filtered_index = []
            # for i in range(NROWS):
            #     raw_eegs = data_tab[i]
            #     self.transform.fit(raw_eegs)
            # for i in range(NROWS):
            #     raw_eegs = data_tab[i]
            #     result = np.stack(self.transform.transform(raw_eegs))
            #     filtered_eeg.append(result)
            #     filtered_label.append(label_tab[i])
            #     filtered_index.append(index_tab[i])

            # if list_ind_to_remove:
            #     filtered_eeg = np.delete(
            #         np.stack(filtered_eeg), list_ind_to_remove, axis=0
            #     )
            #     filtered_label = np.delete(
            #         np.stack(filtered_label), list_ind_to_remove, axis=0
            #     )
            #     filtered_index = np.delete(
            #         np.stack(filtered_index), list_ind_to_remove, axis=0
            #     )
            # else:
            #     filtered_eeg = np.stack(filtered_eeg)
            #     filtered_label = np.stack(filtered_label)
            #     filtered_index = np.stack(filtered_index)

            # output = (
            #     torch.as_tensor(filtered_eeg).float(),
            #     torch.as_tensor(filtered_label).float(),
            #     torch.as_tensor(np.array([index])),
            # )

            # self.cached_data[index] = output

            # return output

        elif self.use_cache and self.cached_data[index] is not None:
            output = self.cached_data[index]
            return output

        elif not self.use_cache:

            output = self._internal_get_data(index)

            return output
            # record = self._read_hdf(self.root / self.meta.loc[index, "filename"])

            # runs_raw = {}
            # for i, act in enumerate(record):
            #     # target and stims are increased by 1
            #     # because the channel is filled with zeros by default
            #     target = act["target"] + 1
            #     run_data = []
            #     for eeg, starts, stims in act["sessions"]:
            #         starts = starts * self.sampling_rate / self._ms_in_sec
            #         starts = starts.round().astype(np.int)
            #         stims = stims + 1
            #         stims_channel = np.zeros(eeg.shape[1])
            #         target_channel = np.zeros(eeg.shape[1])

            #         for start, stimul in zip(starts, stims):
            #             stims_channel[start] = stimul
            #             target_channel[start] = 1 if stimul == target else 2

            #         round_data = np.vstack(
            #             (eeg, stims_channel[None, :], target_channel[None, :])
            #         )
            #         run_data.append(round_data)

            #     raw = np.hstack(run_data)
            #     runs_raw[f"run_{i}"] = raw
            # data = {"session_0": runs_raw}

            # # NROWS : int, number of rows in the dataset output by the loader
            # # 250  = 5 events par period x  10 periods per run x 5 runs with events "1"
            # # runs with no "1" events are not included and used
            # NROWS = 250
            # nrows = NROWS

            # data_tab, label_tab, index_tab = self._get_data_tab(data, nrows)

            # # list of outliers
            # if self.remove_inds is not None:
            #     list_indice = list(self.remove_inds.keys())
            #     list_ind_to_remove = [
            #         int(list_indice[j]) for j in range(len(list_indice))
            #     ]
            # else:
            #     list_ind_to_remove = []

            # # Transforms :
            # filtered_eeg = []
            # filtered_label = []
            # filtered_index = []
            # for i in range(NROWS):
            #     raw_eegs = data_tab[i]
            #     self.transform.fit(raw_eegs)
            # for i in range(NROWS):
            #     raw_eegs = data_tab[i]
            #     result = np.stack(self.transform.transform(raw_eegs))
            #     filtered_eeg.append(result)
            #     filtered_label.append(label_tab[i])
            #     filtered_index.append(index_tab[i])

            # if list_ind_to_remove:
            #     filtered_eeg = np.delete(
            #         np.stack(filtered_eeg), list_ind_to_remove, axis=0
            #     )
            #     filtered_label = np.delete(
            #         np.stack(filtered_label), list_ind_to_remove, axis=0
            #     )
            #     filtered_index = np.delete(
            #         np.stack(filtered_index), list_ind_to_remove, axis=0
            #     )
            # else:
            #     filtered_eeg = np.stack(filtered_eeg)
            #     filtered_label = np.stack(filtered_label)
            #     filtered_index = np.stack(filtered_index)

            # output = (
            #     torch.as_tensor(filtered_eeg).float(),
            #     torch.as_tensor(filtered_label).float(),
            #     torch.as_tensor(np.array([index])),
            # )

            # self.cached_data[index] = output

            # return output

    def _internal_get_data(self, index):

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

        # NROWS : int, number of rows in the dataset output by the loader
        # 250  = 5 events par period x  10 periods per run x 5 runs with events "1"
        # runs with no "1" events are not included and used
        NROWS = 250
        nrows = NROWS

        data_tab, label_tab, index_tab = self._get_data_tab(data, nrows)

        # list of outliers
        if (self.remove_inds is not None) and (str(index) in self.remove_inds.keys()):
            list_indice = list(self.remove_inds[str(index)])
            list_ind_to_remove = [int(list_indice[j]) for j in range(len(list_indice))]
        else:
            list_ind_to_remove = []

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

        if list_ind_to_remove:
            filtered_eeg = np.delete(np.stack(filtered_eeg), list_ind_to_remove, axis=0)
            filtered_label = np.delete(
                np.stack(filtered_label), list_ind_to_remove, axis=0
            )
            filtered_index = np.delete(
                np.stack(filtered_index), list_ind_to_remove, axis=0
            )
        else:
            filtered_eeg = np.stack(filtered_eeg)
            filtered_label = np.stack(filtered_label)
            filtered_index = np.stack(filtered_index)

        output = (
            torch.as_tensor(filtered_eeg).float(),
            torch.as_tensor(filtered_label).float(),
            torch.as_tensor(np.array([index])),
        )

        self.cached_data[index] = output

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

                # find event of interest : label = '1'
                signal = data[f"session_{i}"][f"run_{j}"][9]
                signal_test = [
                    (indice, signal[indice])
                    for indice in range(len(signal))
                    if signal[indice] == 1 or signal[indice] == 2
                ]
                # for each event create one epoch of self.sample_per_batch samples (window size)
                if signal_test == []:
                    continue

                if signal_test != [] and (1 in signal):
                    for k in range(len(signal_test)):

                        # if target then signal = 1  -> label = 1
                        # if not target then signal = 2 -> label = 0

                        if signal_test[k][1] == 0:
                            continue

                        elif signal_test[k][1] == 1 and signal_test[k][
                            0
                        ] + self.sample_per_epoch < len(signal):
                            data_tmp.append(
                                data[f"session_{i}"][f"run_{j}"][
                                    0:8,
                                    signal_test[k][0] : signal_test[k][0]
                                    + self.sample_per_epoch,
                                ]
                            )

                            label_tmp.append(np.array([1.0]))

                            index_tmp.append(
                                np.array(
                                    [
                                        signal_test[k][0],
                                        signal_test[k][0] + self.sample_per_epoch,
                                    ]
                                )
                            )

                        elif signal_test[k][1] == 2 and signal_test[k][
                            0
                        ] + self.sample_per_epoch < len(signal):
                            data_tmp.append(
                                data[f"session_{i}"][f"run_{j}"][
                                    0:8,
                                    signal_test[k][0] : signal_test[k][0]
                                    + self.sample_per_epoch,
                                ]
                            )

                            label_tmp.append(np.array([0.0]))

                            index_tmp.append(
                                np.array(
                                    [
                                        signal_test[k][0],
                                        signal_test[k][0] + self.sample_per_epoch,
                                    ]
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
