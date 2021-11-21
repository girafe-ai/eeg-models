from moabb.datasets import bi2013a

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


class BrainInvadersDataset(AbstractEegDataset):
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        m_dataset = bi2013a(
            NonAdaptive=True,
            Adaptive=True,
            Training=True,
            Online=True,
        )

        m_dataset.download()

        self.data = m_dataset.get_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return {"subject": index, "data": self.data[index]}

    @property
    def channels(self) -> List[str]:
        return self.data[1]["session_1"]["run_1"].ch_names[:-1]

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
