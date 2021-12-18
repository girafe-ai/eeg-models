import glob
import os
import zipfile

import mne
import yaml
from mne.channels import make_standard_montage
from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


BI2013a_URL = "https://zenodo.org/record/1494240/files/"


class bi2013a(BaseDataset):
    def __init__(self, NonAdaptive=True, Adaptive=False, Training=True, Online=False):
        super().__init__(
            subjects=list(range(1, 24 + 1)),
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=2),
            code="Brain Invaders 2013a",
            interval=[0, 1],
            paradigm="p300",
            doi="",
        )

        self.adaptive = Adaptive
        self.nonadaptive = NonAdaptive
        self.training = Training
        self.online = Online

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

            raw_original = mne.io.read_raw_gdf(file_path, preload=True)
            raw_original.rename_channels({"FP1": "Fp1", "FP2": "Fp2"})
            raw_original.set_montage(make_standard_montage("standard_1020"))

            sessions[session_name][run_name] = raw_original

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # check if has the .zip
        url = "{:s}subject{:d}.zip".format(BI2013a_URL, subject)
        path_zip = dl.data_dl(url, "BRAININVADERS")
        path_folder = path_zip.strip("subject{:d}.zip".format(subject))

        # check if has to unzip
        if not (os.path.isdir(path_folder + "subject{:d}".format(subject))):
            print("unzip", path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        # filter the data regarding the experimental conditions
        meta_file = os.path.join("subject{:d}".format(subject), "meta.yml")
        meta_path = path_folder + meta_file
        with open(meta_path, "r") as stream:
            meta = yaml.load(stream, Loader=yaml.FullLoader)
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

        # list the filepaths for this subject
        subject_paths = []
        for filename in filenames:
            subject_paths = subject_paths + glob.glob(
                os.path.join(
                    path_folder, "subject{:d}".format(subject), "Session*", filename
                )
            )  # noqa
        return subject_paths
