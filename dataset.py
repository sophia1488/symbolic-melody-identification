from torch.utils.data import Dataset

class PianorollDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.melody = y

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, index):
        return self.data[index], self.melody[index]
