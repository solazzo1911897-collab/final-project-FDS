import os
import warnings
import random
import numpy as np

# --- NUCLEAR WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*") 

# --- IMPORTS ---
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.dataset import create_dataloaders
from src.model import GWClassifier
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
LABELS_FILE = os.path.join(DATA_DIR, "subset_labels.csv")
MODEL_PATH = "models/best_model.pth"
SEED = 42  # MUST BE THE SAME AS TRAINING

def set_seed(seed):
    """Locks all sources of randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

def generate_advanced_plots():
    # Set Seed (CRITICAL for Data Split Consistency)
    set_seed(SEED)
    
    # Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Device: MacOS GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using Device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using Device: CPU")

    print(f"🚀 Loading Best Model from {MODEL_PATH}...")
    
    # Load Data (Validation Only)
    # Note: create_dataloaders will use the same seed to recreate the same 80/20 split
    _, val_loader = create_dataloaders(DATA_DIR, LABELS_FILE, batch_size=32)
    
    # Load Model
    model = GWClassifier(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Run Inference
    print("🔍 Running Inference on Validation Set...")
    all_preds = []
    all_targets = []
    top_hits = [] 
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images).squeeze()
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
            # Save best hits for gallery
            for i, p in enumerate(preds):
                if targets[i] == 1 and p > 0.9:
                    top_hits.append((p, images[i].cpu(), targets[i]))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # --- PLOT 1: CONFUSION MATRIX ---
    binary_preds = (all_preds > 0.5).astype(int)
    cm = confusion_matrix(all_targets, binary_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Noise', 'GW Signal'],
                yticklabels=['Noise', 'GW Signal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plot_confusion_matrix.png')
    print("✅ Saved plot_confusion_matrix.png")
    
    # --- PLOT 2: ROC CURVE ---
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_roc_curve.png')
    print("✅ Saved plot_roc_curve.png")

    # --- PLOT 3: THE GALAXY GALLERY ---
    top_hits.sort(key=lambda x: x[0], reverse=True)
    best_6 = top_hits[:6]
    
    if len(best_6) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Top Detected Gravitational Waves (Confidence > 90%)", fontsize=16)
        
        for idx, (conf, img_tensor, target) in enumerate(best_6):
            if idx >= 6: break
            row = idx // 3
            col = idx % 3
            
            # Show 1st Channel (LIGO Hanford)
            img_display = img_tensor[0].numpy()
            
            ax = axes[row, col]
            im = ax.imshow(img_display, origin='lower', aspect='auto', cmap='inferno')
            ax.set_title(f"Confidence: {conf*100:.2f}%")
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig('plot_galaxy_gallery.png')
        print("✅ Saved plot_galaxy_gallery.png")
    else:
        print("⚠️ No high-confidence hits found.")

if __name__ == "__main__":
    generate_advanced_plots()