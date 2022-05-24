import glob
import os

import mne
from somepytools.io import read_yaml
from somepytools.typing import Any, Callable, Dict, Directory, Optional, Sequence

from eeg_models import data_dir
from eeg_models.datasets.abstract import AbstractEegDataset


class BrainInvadersDataset(AbstractEegDataset):
    """BrainInvadersDataset dataset from a "Brain Invaders" experiment (2013)
    carried-out at University of Grenoble Alpes.
    This dataset concerns an experiment carried out at GIPSA-lab
    (University of Grenoble Alpes, CNRS, Grenoble-INP) in 2013.For more inf:
    https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/braininvaders.py

    The recordings concerned 24 subjects in total. Subjects 1 to 7 participated
    to eight sessions, run in different days, subject 8 to 24 participated to
    one session. Each session consisted in two runs, one in a Non-Adaptive
    (classical) and one in an Adaptive (calibration-less) mode of operation.

    Data were acquired with a Nexus (TMSi, The Netherlands) EEG amplifier:
    - Sampling Frequency: 512 samples per second
    - Digital Filter: no
    - Electrodes:  16 wet Silver/Silver Chloride electrodes positioned at
      FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2
      according to the 10/20 international system.
    """

    _default_root = "bi2013a"
    subjects = tuple(range(1, 24 + 1))

    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        non_adaptive: bool = True,
        adaptive: bool = True,
        training: bool = True,
        online: bool = True,
    ) -> None:
        self.adaptive = adaptive
        self.non_adaptive = non_adaptive
        self.training = training
        self.online = online

        if root is None:
            root = data_dir / self._default_root

        super().__init__(root, split, transforms, transform, target_transform, download)

    def get_data(self, _subjects: tuple = None) -> Any:
        data = []

        if _subjects is None:
            _subjects = self.subjects

        data = dict()
        for subject in _subjects:
            if subject not in self.subjects:
                raise ValueError("Invalid subject {:d} given".format(subject))
            data[subject] = self._get_single_subject_data(subject)

        return data

    def _get_single_subject_data(self, subject: int) -> Any:
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path in file_path_list:

            session_number = file_path.split(os.sep)[-2].replace("Session", "")
            session_name = "session_" + session_number
            if session_name not in sessions.keys():
                sessions[session_name] = {}

            run_number = file_path.split(os.sep)[-1]
            run_number = run_number.split("_")[-1]
            run_number = run_number.split(".gdf")[0]
            run_name = "run_" + run_number

            raw_original = mne.io.read_raw_edf(
                file_path, montage="standard_1020", preload=True
            )

            sessions[session_name][run_name] = raw_original

        return sessions

    def data_path(
        self,
        subject: int,
    ) -> Optional[Directory]:

        if subject not in self.subjects:
            raise (ValueError("Invalid subject number"))

        meta_file = os.path.join("subject{:d}".format(subject), "meta.yml")
        meta = read_yaml(self.root / meta_file)
        conditions = []
        if self.adaptive:
            conditions = conditions + ["adaptive"]
        if self.non_adaptive:
            conditions = conditions + ["non_adaptive"]
        types = []
        if self.training:
            types = types + ["training"]
        if self.online:
            types = types + ["online"]
        filenames = []
        for run in meta["runs"]:
            run_condition = run["experimental_condition"]
            run_type = run["type"]
            if (run_condition in conditions) and (run_type in types):
                filenames = filenames + [run["filename"]]

        subject_paths = []
        for filename in filenames:
            subject_paths = subject_paths + glob.glob(
                os.path.join(
                    self.root, "subject{:d}".format(subject), "Session*", filename
                )
            )
        return subject_paths

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sessions = self._get_single_subject_data(index)
        eegs, markers = [], []
        for _, run in sorted(sessions["session_1"].items()):
            r_data = run.get_data()
            eegs.append(r_data[:-1])
            markers.append(r_data[-1])
        return {"eegs": eegs, "markers": markers}

    @property
    def channels(self) -> Sequence[str]:
        return self._get_single_subject_data(1)["session_1"]["run_1"].ch_names[:-1]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of original dataset, Hertz"""
        return 512.0

    @property
    def urls(self) -> Dict[str, Sequence[str]]:
        """Gives all known source url sets for this dataset"""
        return {"origin": ["https://doi.org/10.5281/zenodo.1494163"]}

    def download(self):
        pass  # nothing to do - this is DVC based dataset
