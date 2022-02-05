from eeg_models.types import Any, Callable, Dict, Directory, List, Optional


class AbstractEegDataset:
    def __init__(
        self,
        subjects: list = None,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        try:
            _ = iter(subjects)
        except TypeError:
            raise ValueError("subjects must be a iterable, like a list") from None

        self.subject_list = subjects
        self.root = root
        self.split = split
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

    def get_data(self, subjects=None):
        data = []

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

    def download(self, path=None, force_update=False, update_path=None, verbose=None):
        for subject in self.subject_list:
            self.data_path(
                subject=subject,
                path=path,
                force_update=force_update,
                update_path=update_path,
                verbose=verbose,
            )

    def _get_single_subject_data(self, subject):
        pass

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        pass

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def channels(self) -> List[str]:
        raise NotImplementedError()
