from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import numpy as np

class UnpairedDataset(Dataset):

    def __init__(self, split: str):

        # check if train / val or test
        assert split in ["train", "test"], "split must be either 'train', 'val' or 'test'"

        self.horse = load_dataset("gigant/horse2zebra", name="horse", split=split)
        self.zebra = load_dataset("gigant/horse2zebra", name="zebra", split=split)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def __len__(self):
        return max(len(self.horse), len(self.zebra))
    
    def normalize(self, x: torch.Tensor):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def __getitem__(self, idx):
        horse, zebra = self.horse[idx % len(self.horse)], self.zebra[idx % len(self.zebra)]

        (x, y) = (horse["image"], zebra["image"])

        x = torch.tensor(np.array(x)).permute(2, 0, 1).float()
        y = torch.tensor(np.array(y)).permute(2, 0, 1).float()

        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=224, mode="bilinear", align_corners=False).squeeze(0)
        y = torch.nn.functional.interpolate(y.unsqueeze(0), size=224, mode="bilinear", align_corners=False).squeeze(0)

        x = self.normalize(x / 255.0)
        y = self.normalize(y / 255.0)

        return (x, y)
    
# if __name__ == "__main__":

datasets = UnpairedDataset("test")
datasets[0]