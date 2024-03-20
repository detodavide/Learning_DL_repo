import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence


class VariableSizeDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.as_tensor(s).float() for s in x]
        self.y = torch.as_tensor(y).float().view(-1, 1)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

    @staticmethod
    def pack_collate(batch):
        X = [item[0] for item in batch]
        Y = [item[1] for item in batch]

        # False -> Means not sorted sequences by length
        X_pack = pack_sequence(X, enforce_sorted=False)

        return X_pack, torch.as_tensor(Y).view(-1, 1)
