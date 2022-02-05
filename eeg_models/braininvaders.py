import glob
import os
import zipfile

import mne
import yaml

from eeg_models.abstract import AbstractEegDataset
from eeg_models.types import Any, Callable, Dict, Directory, List, Optional
from eeg_models.utils import dt_path


class BrainInvadersDataset(AbstractEegDataset):
    def __init__(
        self,
        subjects: tuple = tuple(range(1, 24 + 1)),
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        NonAdaptive: bool = True,
        Adaptive: bool = True,
        Training: bool = True,
        Online: bool = True,
    ) -> None:

        self.subjects = subjects
        self.adaptive = Adaptive
        self.nonadaptive = NonAdaptive
        self.training = Training
        self.online = Online

        super().__init__(
            subjects, root, split, transforms, transform, target_transform, download
        )

        self.m_data = self.get_data()

        self._data = []
        for _, sessions in sorted(self.m_data.items()):
            eegs, markers = [], []
            for _, run in sorted(sessions["session_1"].items()):
                r_data = run.get_data()
                eegs.append(r_data[:-1])
                markers.append(r_data[-1])
            self._data.append((eegs, markers))

    def _get_single_subject_data(self, subject):
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
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}subject{:d}.zip".format(
            "https://zenodo.org/record/1494240/files/", subject
        )
        path_zip = dt_path(url, "BRAININVADERS")
        path_folder = path_zip.strip("subject{:d}.zip".format(subject))

        if not (os.path.isdir(path_folder + "subject{:d}".format(subject))):
            print("unzip", path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        meta_file = os.path.join("subject{:d}".format(subject), "meta.yml")
        meta_path = path_folder + meta_file
        with open(meta_path, "r") as stream:
            meta = yaml.load(stream)
        conditions = []
        if self.adaptive:
            conditions = conditions + ["adaptive"]
        if self.nonadaptive:
            conditions = conditions + ["nonadaptive"]
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
                    path_folder, "subject{:d}".format(subject), "Session*", filename
                )
            )  # noqa
        return subject_paths

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"eegs": self._data[index][0], "markers": self._data[index][1]}

    @property
    def channels(self) -> List[str]:
        return self.m_data[1]["session_1"]["run_1"].ch_names[:-1]
