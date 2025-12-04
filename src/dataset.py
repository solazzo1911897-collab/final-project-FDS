import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from src.transforms import GWTransform

class G2NetDataset(Dataset):
    def __init__(self, file_paths, targets, training=False):
        self.file_paths = file_paths
        self.targets = targets
        self.training = training
        self.transform = GWTransform()
        self.resize = Resize((224, 224), antialias=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load Data
        path = self.file_paths[idx]
        waves = np.load(path)
        
        # 2. To Tensor
        wave_tensor = torch.from_numpy(waves).float()
        
        # 3. SAFE Normalization (Min-Max per channel)
        for i in range(3):
            w_min = wave_tensor[i].min()
            w_max = wave_tensor[i].max()
            wave_tensor[i] = (wave_tensor[i] - w_min) / (w_max - w_min + 1e-8)

        # 4. CQT & Log Scale
        image = self.transform(wave_tensor, training=self.training)
        
        # 5. Resize to 224x224 (Standard for EfficientNet)
        image = self.resize(image)
        
        # 6. Final Normalization for the Image (0 to 1)
        img_min = image.min()
        img_max = image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)
        
        label = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, label

def prepare_data_lists(data_dir, labels_file):
    """
    NEW FUNCTION: Loads all file paths and targets once.
    This was moved from the old create_dataloaders.
    """
    print(f"Scanning {data_dir} for .npy files...")
    df = pd.read_csv(labels_file)
    
    file_path_map = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npy"):
                file_id = os.path.splitext(file)[0]
                file_path_map[file_id] = os.path.join(root, file)
    
    file_paths = []
    valid_indices = []
    for idx, row in df.iterrows():
        f_id = row['id']
        if f_id in file_path_map:
            file_paths.append(file_path_map[f_id])
            valid_indices.append(idx)
            
    # targets is a NumPy array (crucial for StratifiedKFold)
    targets = df.loc[valid_indices, 'target'].values 
    print(f"Matched {len(file_paths)} samples.")
    
    if len(file_paths) == 0: raise FileNotFoundError("No files found.")
    
    return file_paths, targets


def create_dataloaders(file_paths, targets, train_indices, val_indices, batch_size=32):
    """
    MODIFIED FUNCTION: Accepts pre-split indices from main.py's K-Fold loop.
    """
    
    # Select the paths and targets based on the indices for the current fold
    train_paths = [file_paths[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    val_paths = [file_paths[i] for i in val_indices]
    val_targets = [targets[i] for i in val_indices]
    
    train_dataset = G2NetDataset(train_paths, train_targets, training=True)
    val_dataset = G2NetDataset(val_paths, val_targets, training=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader