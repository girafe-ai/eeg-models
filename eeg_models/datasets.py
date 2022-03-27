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
        self.m_dataset = bi2013a(
            NonAdaptive=True,
            Adaptive=True,
            Training=True,
            Online=True,
        )

        super().__init__(root, split, transforms, transform, target_transform, download)

        self.m_data = self.m_dataset.get_data()

        self.data = []
        for _, sessions in sorted(self.m_data.items()):
            eegs, markers = [], []
            for _, run in sorted(sessions["session_1"].items()):
                r_data = run.get_data()
                eegs.append(r_data[:-1])
                markers.append(r_data[-1])
            self.data.append((eegs, markers))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"eegs": self.data[index][0], "markers": self.data[index][1]}

    @property
    def channels(self) -> List[str]:
        return self.m_data[1]["session_1"]["run_1"].ch_names[:-1]

    def download(self):
        self.m_dataset.download()



class DemonsP300Dataset(AbstractEegDataset):
    def __init__(
        self,
        root: Optional[Directory] = None,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.m_dataset = DemonsP300()

        super().__init__(root, split, transforms, transform, target_transform, download)

        self.m_data = self.m_dataset.get_data()

        self.data = []
        for _, sessions in sorted(self.m_data.items()):
            eegs, markers = [], []
            for _, run in sorted(sessions["session_0"].items()):
                r_data = run.get_data()
                eegs.append(r_data[:-2])
                markers.append(r_data[-2:-1])
            self.data.append((eegs, markers))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"eegs": self.data[index][0], "markers": self.data[index][1]}

    @property
    def channels(self) -> List[str]:
        return self.m_data[1]["session_0"]["run_0"].ch_names[:-2]

    def download(self):
        self.m_dataset.download()


