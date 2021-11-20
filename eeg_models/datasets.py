import moabb.datasets
import torch

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

    def download(self):
        raise NotImplementedError()

    @property
    def channels(self) -> List[str]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError()


class BrainInvadersDataset(AbstractEegDataset):
    def __init__(self):
        m_dataset = moabb.datasets.bi2013a(
            NonAdaptive=True,
            Adaptive=True,
            Training=True,
            Online=True,
        )

        m_dataset.download()

        self.data = m_dataset.get_data()

    def raw_dataset(self) -> List:
        raw_dataset = []
        for _, sessions in sorted(self.data.items()):
            eegs, markers = [], []
            for _item, run in sorted(sessions["session_1"].items()):
                data = run.get_data()
                eegs.append(data[:-1])
                markers.append(data[-1])
            raw_dataset.append((eegs, markers))
        return raw_dataset

    def channels(self) -> List:
        return self.data[1]["session_1"]["run_1"].ch_names[:-1]

    def __getitem__(self, item_idx: int) -> dict:
        self.current_subject = item_idx
        self.current_data = self.data[item_idx]
        return {"subject": self.current_subject, "data": self.current_data}

    def __len__(self) -> int:
        return len(self.data)
