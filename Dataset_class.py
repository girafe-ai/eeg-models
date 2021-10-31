import torch


class Dataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, item_idx):
        current_sample = self.data[item_idx, :]
        current_target = self.targets[item_idx]
        return {
            "sample": torch.tensor(current_sample),
            "target": torch.tensor(current_target)
        }

    def __len__(self):
        return len(self.data)
