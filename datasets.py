import torch
import moabb.datasets


m_dataset = moabb.datasets.bi2013a(
    NonAdaptive=True,
    Adaptive=True,
    Training=True,
    Online=True,
)

m_dataset.download()

m_data = m_dataset.get_data()


class Dataset:  
    def __init__(self, m_data):        
        self.m_data = m_data       
        
    def raw_dataset(self):
        raw_dataset = []
        for _, sessions in sorted(self.m_data.items()):
            eegs, markers = [], []
            for item, run in sorted(sessions['session_1'].items()):
                data = run.get_data()
                eegs.append(data[:-1])
                markers.append(data[-1])
        raw_dataset.append((eegs, markers))        
        return raw_dataset
    
    def channels(self):
        return self.m_data[1]['session_1']['run_1'].ch_names[:-1]        

    def __getitem__(self, item_idx: int):
        current_sample = self.m_data[item_idx]        
        return torch.tensor(current_sample)  

    def __len__(self) -> int:
        return len(self.m_data)


class Dataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, item_idx: int) -> dict:
        current_sample = self.data[item_idx, :]
        current_target = self.targets[item_idx]
        return {
            "sample": torch.tensor(current_sample),
            "target": torch.tensor(current_target)
        }

    def __len__(self) -> int:
        return len(self.data)
