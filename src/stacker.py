import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
DRIVE_PATH = '/content/drive/MyDrive/FDSstacker_input' 

OOF_FILE_NAMES = [
    'oof_preds_efficientnet_b0.npy',
    'oof_preds_efficientnet_b5.npy',
    'oof_preds_efficientnet_b6.npy',
    'oof_preds_efficientnet_b7.npy',
    'oof_preds_efficientnet_b8.npy',
]
OOF_FILES = [os.path.join(DRIVE_PATH, name) for name in OOF_FILE_NAMES]


LABELS_FILE = "data/raw/subset_labels.csv"
SEED = 42
N_STACK_FOLDS = 5
NN_EPOCHS = 50 
NN_LR = 1e-3
BATCH_SIZE = 128
INPUT_DIM = len(OOF_FILES) # 5 features (one from each EfficientNet)

# --- 1. DEFINE THE NEURAL NETWORK (META-MODEL) ---
class MetaNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# --- 2. DATA LOADING AND TRAINING FUNCTION ---
def train_stacker_nn():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n--- Loading Data and Combining Features ---")
    
    # Load True Target Labels (Y_TRUE)
    try:
        df = pd.read_csv(LABELS_FILE)
        Y_TRUE = df['target'].values 
    except FileNotFoundError:
        print(f"❌ Error: Labels file not found at {LABELS_FILE}. Check path.")
        return
    X_list = []
    for filename in OOF_FILES:
        try:
            preds = np.load(filename) 
            X_list.append(preds.reshape(-1, 1))
            print(f"Loaded feature from: {filename}")
        except FileNotFoundError:
            print(f"❌ CRITICAL ERROR: OOF file not found at {filename}. Missing features for stacking.")
            return

    X_stack = np.hstack(X_list).astype(np.float32)
    
    print(f"Stacked Feature Matrix (X_stack) Shape: {X_stack.shape}")
    print(f"True Labels (Y_TRUE) Shape: {Y_TRUE.shape}")
    
    
    print("\n--- 3. Training PyTorch Meta-Model ---")
    
    kf = KFold(n_splits=N_STACK_FOLDS, shuffle=True, random_state=SEED)
    meta_oof_predictions = np.zeros(X_stack.shape[0])
    cv_scores = []
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_stack, Y_TRUE)):
        print(f"\n======== FOLD {fold+1}/{N_STACK_FOLDS} ========")
        
        X_train_tensor = torch.from_numpy(X_stack[train_idx])
        Y_train_tensor = torch.from_numpy(Y_TRUE[train_idx]).float()
        X_val_tensor = torch.from_numpy(X_stack[val_idx])
        Y_val_tensor = torch.from_numpy(Y_TRUE[val_idx]).float()

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize Model and Optimizer
        meta_model = MetaNN(input_dim=INPUT_DIM).to(device)
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=NN_LR)
        
        for epoch in range(NN_EPOCHS):
            meta_model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = meta_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        meta_model.eval()
        all_val_preds = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                outputs = meta_model(x_batch)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_val_preds.extend(preds)
        
        val_preds_np = np.array(all_val_preds)
        meta_oof_predictions[val_idx] = val_preds_np
        
        fold_auc = roc_auc_score(Y_TRUE[val_idx], val_preds_np)
        cv_scores.append(fold_auc)
        print(f"  -> Fold {fold+1} Finished. Val AUC: {fold_auc:.6f}")

    # 4. FINAL RESULTS
    print("\n" + "="*50)
    print("🏁 FINAL STACKING RESULTS (PyTorch NN Meta-Model)")
    print(f"Single Stacked AUC: {roc_auc_score(Y_TRUE, meta_oof_predictions):.6f}")
    print(f"Mean CV AUC:        {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print("="*50)


if __name__ == "__main__":
    train_stacker_nn()