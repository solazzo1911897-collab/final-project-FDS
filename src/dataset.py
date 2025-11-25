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

def create_dataloaders(data_dir, labels_file, batch_size=32, split_ratio=0.8):
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
            
    targets = df.loc[valid_indices, 'target'].values
    print(f"Matched {len(file_paths)} samples.")
    
    if len(file_paths) == 0: raise FileNotFoundError("No files found.")

    dataset_size = len(file_paths)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = G2NetDataset([file_paths[i] for i in train_idx], [targets[i] for i in train_idx], training=True)
    val_dataset = G2NetDataset([file_paths[i] for i in val_idx], [targets[i] for i in val_idx], training=False)
    
    # DataLoader Standard (Without explicit num_workers or pin_memory)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader