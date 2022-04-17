from eeg_models.types import Any, Callable, Dict, Directory, List, Optional


class AbstractEegDataset:
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def channels(self) -> List[str]:
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()

