import os
import warnings
import random
import numpy as np

# --- 1. NUCLEAR WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*") 
#comment
# --- 2. IMPORTS ---
import torch
import matplotlib.pyplot as plt
from src.dataset import prepare_data_lists, create_dataloaders
from src.model import GWClassifier
from src.train import Trainer
from sklearn.model_selection import StratifiedKFold

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
LABELS_FILE = os.path.join(DATA_DIR, "subset_labels.csv")
MODEL_SAVE_PATH = "models/best_model.pth"

BATCH_SIZE = 32
EPOCHS = 12   
LEARNING_RATE = 5e-5 
SEED = 42 
K_FOLDS = 5        
MODEL_BACKBONE = 'efficientnet_b0'
def set_seed(seed):
    """
    Locks all sources of randomness to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

def plot_results(history, fold_num=None): 
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: AUC Score
    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC', linestyle='--')
    plt.title('AUC Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    suffix = f"_fold_{fold_num}" if fold_num else ""
    plt.savefig(f"training_results{suffix}.png")
    print(f"\n📊 Plots saved as 'training_results{suffix}.png'")

def main():
    # 0. Set Seed
    set_seed(SEED)

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Device: MacOS GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using Device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using Device: CPU")

    # 2. Data preparation
    print("\n[1/4] Preparing ALL Data Lists...")
    try:
        all_file_paths, all_targets = prepare_data_lists(DATA_DIR, LABELS_FILE)
        print(f"Total samples: {len(all_file_paths)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

   # 3. K-Fold Setup
    print(f"\n[2/4] Initializing {K_FOLDS}-Fold Stratified Cross-Validation...")
    
    # StratifiedKFold 
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    # OOF predictions
    oof_predictions = np.zeros(len(all_file_paths))
    # K-Fold Iteration Loop
    for fold, (train_index, val_index) in enumerate(skf.split(all_file_paths, all_targets)):
        
        print(f"\n\n====================== FOLD {fold+1}/{K_FOLDS} ======================")
        fold_save_path = f"models/best_model_fold_{fold+1}.pth"
        
        # --- A. Create Data Loaders for Current Fold ---
        train_loader, val_loader = create_dataloaders(
            all_file_paths,
            all_targets,
            train_index, 
            val_index,   
            batch_size=BATCH_SIZE
        )
        print(f"Fold {fold+1}: Train samples: {len(train_index)} | Val samples: {len(val_index)}")

        # --- B. Initialize Model for Current Fold ---
        print(f"\n[3/4] Initializing Fresh {MODEL_BACKBONE} Model...")
        model = GWClassifier(
            model_name=MODEL_BACKBONE, 
            pretrained=True
        ).to(device) 
        # --- C. Start Training for Current Fold ---
        print("\n[4/4] Starting Training Loop...")
        os.makedirs("models", exist_ok=True)
        
        #FRESH Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=LEARNING_RATE
        )
        
        history = trainer.fit(epochs=EPOCHS, save_path=fold_save_path)
        # D. Capture OOF Predictions for Stacking
        val_preds = trainer.get_val_predictions()
        oof_predictions[val_index] = val_preds
        print(f"Captured {len(val_preds)} OOF predictions for fold {fold+1}.")
        fold_results.append({
            'fold': fold + 1,
            'best_val_auc': trainer.best_score,
            'save_path': fold_save_path,
            'history': history
        })
        
        # Plot for the current fold
        plot_results(history, fold_num=fold+1)

    # 4. Wrap up and Summarize
    DRIVE_PATH = '/content/drive/MyDrive/FDSstacker_input'
    os.makedirs(DRIVE_PATH, exist_ok=True) 
    
    oof_filename = f"oof_preds_{MODEL_BACKBONE}.npy"
    save_path_full = os.path.join(DRIVE_PATH, oof_filename)
    np.save(save_path_full, oof_predictions)
    
    print(f"\n💾 Saved FINAL OOF Predictions for Stacking to: {save_path_full}")
    print("\n\n✅ K-Fold Training Complete!")
    
    total_best_auc = [res['best_val_auc'] for res in fold_results]
    
    print("\n## 🏁 K-Fold Summary")
    for res in fold_results:
        print(f"Fold {res['fold']}: Best Val AUC = {res['best_val_auc']:.4f}")
    
    avg_auc = np.mean(total_best_auc)
    std_auc = np.std(total_best_auc)
    print(f"\n**Average Best Val AUC across {K_FOLDS} Folds:** ${avg_auc:.4f} \pm {std_auc:.4f}$")


if __name__ == "__main__":
    main()