from eeg_models.datasets import BrainInvadersDataset
from eeg_models.types import Any, Callable, Dict, Directory, Optional


class Bi2013a(BrainInvadersDataset):
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        super().__init__(root, split, transforms, transform, target_transform, download)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"eegs": self.data[index][0], "markers": self.data[index][1]}

    def download(self):
        self.m_dataset.download()
