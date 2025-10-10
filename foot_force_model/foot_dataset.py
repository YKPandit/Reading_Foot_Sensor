import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class FootDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )

        # Build label ↔ index maps
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Collect file paths and their encoded labels
        for label in self.classes:
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                if file.endswith(".csv"):
                    self.samples.append(os.path.join(folder, file))
                    self.labels.append(self.label_to_index[label])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.samples[idx], header=None, names=["time","foot1","foot2","foot3","foot4"])
        # Use only the 4 force columns
        x = df[["foot1", "foot2", "foot3", "foot4"]].values.astype("float32")
        y = self.labels[idx]

        target_len = 90
        cur_len = len(x)

        if cur_len < target_len:
            # Pad with the last row (repeats final reading)
            pad = np.repeat(x[-1][None, :], target_len - cur_len, axis=0)
            x = np.concatenate([x, pad], axis=0)
        elif cur_len > target_len:
            # Trim excess readings
            x = x[:target_len]

        return torch.tensor(x), torch.tensor(y)
