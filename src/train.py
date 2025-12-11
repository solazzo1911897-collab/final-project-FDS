import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import time 

class Trainer:
    """
    Training pipeline for the GW Classifier.
    Includes: Training loop, Validation, Gradient Clipping, LR Scheduling, and Timers.
    """
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer: AdamW is standard for EfficientNet
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        
        # Metrics & History
        self.best_score = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            # 1. Zero Gradients
            self.optimizer.zero_grad()
            
            # 2. Forward Pass
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, targets)
            
            # 3. Backward Pass
            loss.backward()
            
            # 4. Gradient Clipping (Prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 5. Optimizer Step
            self.optimizer.step()
            
            # Stats
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader)
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except:
            epoch_auc = 0.5
            
        return epoch_loss, epoch_auc

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = running_loss / len(self.val_loader)
        try:
            auc_score = roc_auc_score(all_targets, all_preds)
        except:
            auc_score = 0.5
            
        return avg_loss, auc_score
    # Raw predictions
    def get_val_predictions(self):
        self.model.eval()
        all_preds = []
        
        # Validation loader
        with torch.no_grad():
            for images, _ in self.val_loader: 
                images = images.to(self.device)
                
                outputs = self.model(images).squeeze()
                # Move to CPU(to implement numpy)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                
        return np.array(all_preds)

    def fit(self, epochs, save_path="models/best_model.pth"):
        print(f"Starting training on {self.device}...")
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        total_start = time.time() # Start TOTAL timer
        
        for epoch in range(epochs):
            epoch_start = time.time() # Start EPOCH timer
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_auc = self.train_one_epoch()
            val_loss, val_auc = self.evaluate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            scheduler.step(val_auc)
            
            # Calculate Epoch Time
            epoch_end = time.time()
            epoch_mins = int((epoch_end - epoch_start) / 60)
            epoch_secs = int((epoch_end - epoch_start) % 60)
            
            print(f"⏱️ Time: {epoch_mins}m {epoch_secs}s")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
            
            if val_auc > self.best_score:
                print(f"🚀 Score Improved ({self.best_score:.4f} -> {val_auc:.4f}). Saving model...")
                self.best_score = val_auc
                torch.save(self.model.state_dict(), save_path)
            else:
                print("Score did not improve.")
        
        # Calculate Total Training Time
        total_end = time.time()
        total_mins = int((total_end - total_start) / 60)
        total_secs = int((total_end - total_start) % 60)
        print(f"\n🏁 Total Training Time: {total_mins}m {total_secs}s")
        
        return self.history