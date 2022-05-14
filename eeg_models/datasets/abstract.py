from somepytools.typing import Any, Callable, Dict, Directory, Optional, Sequence

from .constants import SPLIT_TRAIN


class AbstractEegDataset:
    """Common interface for any dataset.

    Uses PyTorch convention to implement iterable dataset compatible with DataLoader.

    Args:
        root: path to directory with dataset files. None for default location.
        split: one of `.constans.SPLITS`. Splits assumed to be persistent across class instantinations.
    """

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
    def urls(self) -> Dict[str, Sequence[str]]:
        """Gives all known source url sets for this dataset"""
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()
