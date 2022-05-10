from somepytools.typing import Any, Callable, Dict, Directory, Optional, Sequence

from .constants import SPLIT_TRAIN


class AbstractEegDataset:
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = SPLIT_TRAIN,
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
    def channels(self) -> Sequence[str]:
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of original dataset, Hertz"""
        raise NotImplementedError()

    @property
    def urls(self) -> Sequence[str]:
        """Gives url"""
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()
