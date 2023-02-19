from torch.utils.data import Dataset, DataLoader
import torch


class SimulationDataset(Dataset):
    def __init__(self, list_IDs, labels):
        super().__init__()
        self.labels = labels
        self.list_IDs = list_IDs

    def __getitem__(self, item):
        ID = self.list_IDs[item]

        X = ID + '_yeet'
        y = self.labels[ID]
        return X, y

    def __len__(self):
        return len(self.list_IDs)


if __name__ == '__main__':
    label = {'a': 1, 'b': 1, 'c': 0, 'd': 1}
    partition = {'train': ['a', 'b', 'c'], 'validation': ['d']}
    train_set = SimulationDataset(partition['train'], label)
    val_set = SimulationDataset(partition['validation'], label)
    params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 2,
    }
    train_loader = DataLoader(train_set, **params)
    val_loader = DataLoader(val_set, **params)

    for x, y in train_loader:
        print(x, y)

