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

# --- 2. IMPORTS ---
import torch
import matplotlib.pyplot as plt
from src.dataset import create_dataloaders
from src.model import GWClassifier
from src.train import Trainer

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
LABELS_FILE = os.path.join(DATA_DIR, "subset_labels.csv")
MODEL_SAVE_PATH = "models/best_model.pth"

BATCH_SIZE = 32
EPOCHS = 12   
LEARNING_RATE = 5e-5 
SEED = 42         

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
        # Ensure deterministic behavior on CUDA (might slow down slightly but worth it)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

def plot_results(history):
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
    plt.savefig("training_results.png")
    print("\n📊 Plots saved as 'training_results.png'")

def main():
    # 0. Set Seed (FIRST THING TO DO)
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

    # 2. Prepare Data
    print("\n[1/3] Loading Data (Smart Search)...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=DATA_DIR,
            labels_file=LABELS_FILE,
            batch_size=BATCH_SIZE
        )
        print(f"Data loaded successfully.")
        print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 3. Initialize Model
    print("\n[2/3] Initializing EfficientNet Model (RGB Mode)...")
    model = GWClassifier(pretrained=True)
    
    # 4. Start Training
    print("\n[3/3] Starting Training Loop...")
    os.makedirs("models", exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LEARNING_RATE
    )
    
    history = trainer.fit(epochs=EPOCHS, save_path=MODEL_SAVE_PATH)
    
    # 5. Wrap up
    plot_results(history)
    print(f"\n✅ Training Complete! Best model weights saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()