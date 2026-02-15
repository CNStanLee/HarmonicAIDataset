import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HarmonicDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        split: str = "train",       # "train", "val", "test"
        cycle: str = "half",        # "half", "quarter", "full"
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        """
        npz_path: .npz file containing signals and labels
        split: "train" / "val" / "test"
        cycle: "half" = half cycle (32 points), "quarter" = quarter cycle (16 points),
               "full" = full cycle (64 points)
        train_ratio, val_ratio: train/val ratios; test is 1 - train_ratio - val_ratio
        """
        assert split in ("train", "val", "test")
        assert cycle in ("half", "quarter", "full")
        assert 0.0 < train_ratio < 1.0
        assert 0.0 <= val_ratio < 1.0
        assert train_ratio + val_ratio < 1.0

        data = np.load(npz_path)
        signals = data["signals"].astype(np.float32)
        labels = data["labels"].astype(np.float32)

        num_samples, samples_per_cycle = signals.shape

        # Determine input length by cycle
        if cycle == "quarter":
            input_len = samples_per_cycle // 4
        elif cycle == "half":
            input_len = samples_per_cycle // 2
        else:  # "full"
            input_len = samples_per_cycle

        signals = signals[:, :input_len]

        # Normalize by global max absolute value
        max_abs = np.max(np.abs(signals))
        if max_abs > 0:
            signals = signals / max_abs

        # Scale labels for training convenience
        label_scale = 100.0
        labels = labels / label_scale

        # Split train / val / test
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val

        if split == "train":
            self.signals = signals[:num_train]
            self.labels = labels[:num_train]
        elif split == "val":
            self.signals = signals[num_train:num_train + num_val]
            self.labels = labels[num_train:num_train + num_val]
        else:  # "test"
            self.signals = signals[num_train + num_val:]
            self.labels = labels[num_train + num_val:]

        self.input_len = input_len
        self.label_scale = label_scale
        self.split = split
        self.cycle = cycle

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def create_harmonic_datasets(
    npz_path: str,
    cycle: str = "half",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Create train/val/test datasets in one call.
    """
    train_ds = HarmonicDataset(
        npz_path=npz_path,
        split="train",
        cycle=cycle,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    val_ds = HarmonicDataset(
        npz_path=npz_path,
        split="val",
        cycle=cycle,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_ds = HarmonicDataset(
        npz_path=npz_path,
        split="test",
        cycle=cycle,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return train_ds, val_ds, test_ds
