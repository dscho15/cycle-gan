from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import numpy as np
from torchvision import transforms


class UnpairedDataset(Dataset):

    def __init__(
        self,
        split: str,
        size: int = 256,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        rescale_factor: float = 255.0,
    ):

        # check if train / val or test
        assert split in [
            "train",
            "test",
        ], "split must be either 'train', 'val' or 'test'"

        self.horse = load_dataset("gigant/horse2zebra", name="horse", split=split)
        self.zebra = load_dataset("gigant/horse2zebra", name="zebra", split=split)

        self.mean = (
            torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1) if mean is None else mean
        )
        self.std = (
            torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1) if std is None else std
        )
        self.size = size
        self.rescale_factor = rescale_factor

        self.transforms = transforms.Compose(
            [transforms.Resize((size, size)), transforms.RandomHorizontalFlip()]
        )

    def __len__(self):
        return max(len(self.horse), len(self.zebra))

    def normalize(self, x: torch.Tensor):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def __getitem__(self, idx):
        horse, zebra = (
            self.horse[idx % len(self.horse)],
            self.zebra[idx % len(self.zebra)],
        )

        (x, y) = (horse["image"], zebra["image"])

        x = torch.tensor(np.array(x)).permute(2, 0, 1).float()
        y = torch.tensor(np.array(y)).permute(2, 0, 1).float()

        x = self.normalize(x / self.rescale_factor)
        y = self.normalize(y / self.rescale_factor)

        x = self.transforms(x)
        y = self.transforms(y)

        return (x, y)


# if __name__ == "__main__":

datasets = UnpairedDataset("test")
datasets[0]
