import os
import os.path as osp
import zipfile
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib import parse, request

import h5py
import numpy as np
from pooch import file_hash, retrieve
from somepytools.typing import Directory

from .abstract import AbstractEegDataset
from .constants import SPLIT_TRAIN

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..transforms import ButterFilter, Decimator, ChannellwiseScaler, MarkersTransformer

class DemonsP300Dataset(AbstractEegDataset):
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = SPLIT_TRAIN,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.ch_names = ("Cz", "P3", "Pz", "P4", "PO3", "PO4", "O1", "O2")
        self.sampling_rate = 500.0
        self.url = "https://gin.g-node.org/v-goncharenko/neiry-demons/raw/master/nery_demons_dataset.zip"

        self._ms_in_sec = 1000
        self._hdf_path = "p300dataset"
        self._ds_folder_name = "demons"

        self._act_dtype = np.dtype(
            [
                ("id", np.int),
                ("target", np.int),
                ("is_train", np.bool),
                ("prediction", np.int),
                ("sessions", np.object),  # list of `_session_dtype`
            ]
        )

        self._session_dtype = np.dtype(
            [
                ("eeg", np.object),
                ("starts", np.object),
                ("stimuli", np.object),
            ]
        )

        self.subject_list = list(range(60))
        self.path = None
        self.subjects_filenames = None

        super().__init__(root, split, transforms, transform, target_transform, download)

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

    def read_hdf(self, filename) -> np.ndarray:
        """Reads data from HDF file
        Returns:
            array of `_act_dtype`
        """
        with h5py.File(filename, "r") as hfile:
            group = hfile[self._hdf_path]
            record = np.empty(len(group), self._act_dtype)
            for i, act in enumerate(group.values()):
                record[i]["sessions"] = np.array(
                    [self._strip(item) for item in act], self._session_dtype
                )
                for name, value in act.attrs.items():
                    record[i][name] = value
        return record

    def _get_single_subject_data(self, subject: int):
        record = self.read_hdf(self.data_path(subject))
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
        return {"session_0": runs_raw}

    def data_dl(self, url, sign, path=None, force_update=False, verbose=None):
        path = osp.join(osp.expanduser("~"), "mne_data")
        if not osp.isdir(path):
            os.makedirs(path)

        key_dest = "MNE-{:s}-data".format(sign.lower())

        destination = parse.urlparse(url).path
        if len(destination) < 2 or destination[0] != "/":
            raise ValueError("Invalid URL")
        destination = os.path.join(
            osp.join(path, key_dest), request.url2pathname(destination)[1:]
        )

        # Fetch the file
        if not osp.isfile(destination) or force_update:
            if osp.isfile(destination):
                os.remove(destination)
            if not osp.isdir(osp.dirname(destination)):
                os.makedirs(osp.dirname(destination))
            known_hash = None
        else:
            known_hash = file_hash(destination)
        dlpath = retrieve(
            url,
            known_hash,
            fname=osp.basename(url),
            path=osp.dirname(destination),
            progressbar=True,
        )
        return dlpath

    def data_path(
        self, subject: int, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        zip_path = Path(self.data_dl(self.url, self._ds_folder_name))
        self.path = zip_path.parent / self._ds_folder_name / zip_path.stem

        if not self.path.exists():
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(self.path.parent)

        self.subjects_filenames = sorted(self.path.glob("*.hdf5"))

        return self.subjects_filenames[subject].as_posix()

    def download(
        self,
        subject_list=None,
        path=None,
        force_update=False,
        update_path=None,
        accept=False,
        verbose=None,
    ):
        if subject_list is None:
            subject_list = self.subject_list
        for subject in subject_list:
            # check if accept is needed
            sig = signature(self.data_path)
            if "accept" in [str(p) for p in sig.parameters]:
                self.data_path(
                    subject=subject,
                    path=path,
                    force_update=force_update,
                    update_path=update_path,
                    verbose=verbose,
                    accept=accept,
                )
            else:
                self.data_path(
                    subject=subject,
                    path=path,
                    force_update=force_update,
                    update_path=update_path,
                    verbose=verbose,
                )

    @property
    def channels(self) -> List[str]:
        return self.ch_names

    def __len__(self) -> int:
        return len(self.subject_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self._get_single_subject_data(index)
        
        
        # data_tab = []
        # for i in range(len(data)):
        #     # runs_raw[f"run_{i}"] = raw
        #     for j in range(len(data[f"session_{i}"])):
        #         data_tab.append(data[f"session_{i}"][f"run_{j}"])
        # data_tab = np.hstack(data_tab)
        # return {
        #     "eegs": data_tab[0:8],
        #     "stims_channel": data_tab[8],
        #     "target_channel": data_tab[9],
        # }

        data_tab_max = []
        data_tab_min = []

        
        for i in range(len(data)):
            l_max = max([  data[f"session_{i}"][f"run_{j}"].shape[1] for j in range(len(data[f"session_{i}"]))])
            l_min = min([  data[f"session_{i}"][f"run_{j}"].shape[1] for j in range(len(data[f"session_{i}"]))])
            
            for j in range(len(data[f"session_{i}"])):
                L = l_max - data[f"session_{i}"][f"run_{j}"].shape[1]
                new_run = np.pad(data[f"session_{i}"][f"run_{j}"], ((0,0),(0,L)), mode ='constant', constant_values=((0,0),(0,0)))
                
                data_tab_max.append(new_run)
                data_tab_min.append(data[f"session_{i}"][f"run_{j}"][:, :l_min])

        data_tab_max = np.stack(data_tab_max)     #  ( ndarray : #_runs, #_channels, #_timestamped_data)
        data_tab_min = np.stack(data_tab_min)     #  ( ndarray : #_runs, #_channels, #_timestamped_data)
        

        # Transforms : 
        sampling_rate = 512
        decimation_factor = 1

        for i in range(len(data_tab_max)):
            eeg_pipe = make_pipeline(
                #transforms.Decimator(decimation_factor),
                ButterFilter(sampling_rate // decimation_factor, 4, 0.5, 20),
                ChannellwiseScaler(StandardScaler()),
                )
            eegs = data_tab_max[i][0:8]
            eeg_pipe.fit(eegs)
            data_tab_max[i][0:8] = eeg_pipe.transform(eegs)


        return data_tab_max    #  ( ndarray : #_runs, #_channels, #_timestamped_data)


