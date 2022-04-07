import zipfile
from pathlib import Path
import h5py
import os
from inspect import signature
import os.path as osp
from pooch import file_hash, retrieve
from urllib import parse, request 
from typing import Any, Callable, Dict, List, NewType, Optional
import numpy as np

from .datasets import AbstractEegDataset

Directory = Path


class DemonsP300Dataset(AbstractEegDataset):
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        
        self.ch_names = ["Cz", "P3", "Pz", "P4", "PO3", "PO4", "O1", "O2"]
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

        try:
            _ = iter(self.subject_list)
        except TypeError:
            raise ValueError("subjects must be a iterable, like a list") from None

        self.n_sessions = 1
        self.event_id = {"Target": 1, "NonTarget": 2}
        self.code = "Demons P300"
        self.interval = [0, 1]
        self.paradigm = "p300"
        self.doi = None
        self.unit_factor = 1e6
        self.path = None
        self.subjects_filenames = None

        super().__init__(root, split, transforms, transform, target_transform, download)

        self.m_data = self.get_data()

        self.data = []

        for _, sessions in sorted(self.m_data.items()):
            eegs, markers = [], []

            for _, run in sorted(sessions["session_0"].items()):
                eegs.append(run[:-2])
                markers.append(run[-2:])

            self.data.append((eegs, markers))


    def get_data(self, subjects=None):
        if subjects is None:
            subjects = self.subject_list

        if not isinstance(subjects, list):
            raise (ValueError("subjects must be a list"))

        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError("Invalid subject {:d} given".format(subject))
            data[subject] = self._get_single_subject_data(subject)

        return data


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


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"eegs": self.data[index][0], "markers": self.data[index][1]}

    @property
    def channels(self) -> List[str]:
        return self.ch_names

    def get_dataset_path(self, sign, path):
        sign = sign.upper()
        key = "MNE_DATASETS_{:s}_PATH".format(sign)
        path_def = osp.join(osp.expanduser("~"), "mne_data")
        if not osp.isdir(path_def):
            os.makedirs(path_def)
        return path_def
        

    def data_dl(self, url, sign, path=None, force_update=False, verbose=None):
        path = self.get_dataset_path(sign, path)
        key_dest = "MNE-{:s}-data".format(sign.lower())
        
        destination = parse.urlparse(url).path
        if len(destination) < 2 or destination[0] != '/':
            raise ValueError("Invalid URL")
        destination = os.path.join(osp.join(path, key_dest), request.url2pathname(destination)[1:])

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