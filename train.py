import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from models.resnet18 import ResNetSentinel
from models.shallow import ShallowCNN
import numpy as np
from tqdm import tqdm
import os

# H5 Dataset Class
class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.file = h5py.File(h5_path, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, lbl
    
# Temperature Scaling Module
class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model, dataloader, device):
        """Calibrate temperature parameter using validation data"""
        model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"Temperature scaling calibrated: T = {self.temperature.item():.4f}")

# Class-specific threshold calculation
def best_thresholds_per_class(y_true, y_probs):
    """Calculate optimal threshold per class using F1 score"""
    thresholds = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:  # Handle classes with no positive samples
            thresholds.append(0.5)
            continue

        prec, rec, thresh = precision_recall_curve(y_true[:, i], y_probs[:, i])

        # Handle edge case where thresh is empty
        if len(thresh) == 0:
            thresholds.append(0.5)
            continue

        # Calculate F1 scores
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-6)
        best_idx = np.argmax(f1)
        thresholds.append(thresh[best_idx])

    return np.array(thresholds)

# Training and Evaluation
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.

    for imgs, labels in tqdm(dataloader, desc='Training', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, class_thresholds=None, temperature_scaler=None):
    model.eval()
    running_loss = 0.
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            # Apply temperature scaling if available
            if temperature_scaler is not None:
                outputs = temperature_scaler(outputs)

            # applying sigmoid for multi-label
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Use class-specific thresholds if available
            if class_thresholds is not None:
                preds = (probs > class_thresholds).astype(int)
            else:
                preds = (probs > 0.5).astype(int)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    avg_loss = running_loss / len(dataloader.dataset)

    return avg_loss, f1, precision, recall, acc, all_labels, all_probs

# Main Training loop
def main():
    # path to H5 datasets
    train_path = 'train_data.h5'
    val_path = 'val_data.h5'
    test_path = 'test_data.h5'

    batch_size = 16
    num_epochs = 30 # Total number of epochs
    head_epochs = 10 # Number of epochs to train only the head
    lr_head = 0.001 # Learning rate for head training
    lr_finetune = 0.0001 # Learning rate for fine-tuning

    # Resuming from checkpoint
    resume = False
    checkpoint_path = 'checkpoints/resnet_sentinel_pretrained_epoch15.pth'

    # Early stopping parameters
    early_stop = True
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    start_epoch = 0

    # Datasets and Dataloaders
    train_dataset = H5Dataset(train_path)
    val_dataset = H5Dataset(val_path)
    test_dataset = H5Dataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    num_bands = train_dataset[0][0].shape[0]
    num_classes = train_dataset[0][1].shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = ShallowCNN(num_bands=num_bands, num_classes=num_classes).to(device)
    print('\n' + '='*60)
    print('MODEL INITIALIZATION')
    print('='*60)
    print('🔥 Using ImageNet Pretrained ResNet18')
    print(f'📊 Adapting from 3 RGB channels to {num_bands} spectral bands')
    print(f'🎯 Output classes: {num_classes}')
    print('🔒 Backbone will be frozen during head training phase')
    print('='*60)

    model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=True, freeze_backbone=True).to(device)

    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'📈 Total parameters: {total_params:,}')
    print(f'❄️  Frozen parameters: {frozen_params:,}')
    print(f'🔥 Trainable parameters: {trainable_params:,}')
    print(f'📱 Device: {device}')
    print('='*60)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Value history
    train_loss_history = []
    val_loss_history = []
    val_f1_history = []
    val_prec_history = []
    val_rec_history = []
    val_acc_history = []

    # Initialize temperature scaling and class thresholds
    temperature_scaler = TemperatureScaling().to(device)
    class_thresholds = None

    if resume and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        if start_epoch >= head_epochs:
            optimizer = optim.Adam(model.parameters(), lr=lr_finetune, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        val_f1_history = checkpoint.get('val_f1_history', [])
        val_prec_history = checkpoint.get('val_prec_history', [])
        val_rec_history = checkpoint.get('val_rec_history', [])
        val_acc_history = checkpoint.get('val_acc_history', [])
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    else:
        print("No checkpoint found, starting fresh training")
        start_epoch = 0

    # Phase 1: Head training
    if start_epoch < head_epochs:
        print("Starting head giving training phase")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        for epoch in range(start_epoch, head_epochs):
            print(f'\nEpoch {epoch+1}/{head_epochs}')

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1, val_prec, val_rec, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}')

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_f1_history.append(val_f1)
            val_prec_history.append(val_prec)
            val_rec_history.append(val_rec)
            val_acc_history.append(val_acc)

    # Phase 2: Fine-tuning
    print("Starting fine-tuning phase")
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=lr_finetune, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(max(start_epoch, head_epochs), num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_prec, val_rec, val_acc, val_labels, val_probs = evaluate(model, val_loader, criterion, device, class_thresholds, temperature_scaler)

        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}')

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)
        val_prec_history.append(val_prec)
        val_rec_history.append(val_rec)
        val_acc_history.append(val_acc)

        # Early stopping check
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0

                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                    "val_f1_history": val_f1_history,
                    "val_prec_history": val_prec_history,
                    "val_rec_history": val_rec_history,
                    "val_acc_history": val_acc_history,
                    "best_val_loss": best_val_loss,
                }, f"checkpoints/resnet_best_calibrated.pth")
                print(f"Saved new best model at {epoch + 1}")

            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping niggered")
                    break

    # Post-training calibration and threshold optimization
    print('\n' + '='*50)
    print('POST-TRAINING CALIBRATION & OPTIMIZATION')
    print('='*50)

    # 1. Temperature scaling calibration
    print('\n1. Calibrating temperature scaling on validation set...')
    temperature_scaler.calibrate(model, val_loader, device)

    # 2. Calculate class-specific thresholds
    print('\n2. Computing optimal class-specific thresholds...')
    val_loss, val_f1, val_prec, val_rec, val_acc, val_labels, val_probs = evaluate(model, val_loader, criterion, device, None, temperature_scaler)
    class_thresholds = best_thresholds_per_class(val_labels, val_probs)

    print('Class-specific thresholds:')
    class_names = ['Marine Debris', 'Dense Sargassum', 'Sparse Sargassum', 'Natural Organic Material',
                   'Ship', 'Clouds', 'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water',
                   'Shallow Water', 'Waves', 'Cloud Shadows', 'Wakes', 'Mixed Water']

    for i, (name, thresh) in enumerate(zip(class_names, class_thresholds)):
        print(f'  {name}: {thresh:.3f}')

    # 3. Re-evaluate validation set with calibrated model and optimized thresholds
    print('\n3. Re-evaluating validation set with calibration and optimized thresholds...')
    val_loss_cal, val_f1_cal, val_prec_cal, val_rec_cal, val_acc_cal, _, _ = evaluate(model, val_loader, criterion, device, class_thresholds, temperature_scaler)
    print(f'Calibrated Val Loss: {val_loss_cal:.4f} | Accuracy: {val_acc_cal:.4f} | F1: {val_f1_cal:.4f} | Precision: {val_prec_cal:.4f} | Recall: {val_rec_cal:.4f}')

    # 4. Final test evaluation with all improvements
    print('\n4. Final test evaluation with calibration and optimized thresholds...')
    test_loss, test_f1, test_prec, test_rec, test_acc, _, _ = evaluate(model, test_loader, criterion, device, class_thresholds, temperature_scaler)
    print(f'Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f}')

    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_f1': val_f1_history,
        'val_precision': val_prec_history,
        'val_recall': val_rec_history,
        'val_accuracy': val_acc_history,
    }

    # Save final model checkpoint with calibration
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'temperature_scaler_state_dict': temperature_scaler.state_dict(),
        'class_thresholds': class_thresholds,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_loss_history[-1] if val_loss_history else 0,
        'history': history,
        'calibrated_metrics': {
            'val_f1': val_f1_cal,
            'val_precision': val_prec_cal,
            'val_recall': val_rec_cal,
            'val_accuracy': val_acc_cal,
            'test_f1': test_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_accuracy': test_acc
        }
    }, f"checkpoints/resnet_calibrated_epoch{epoch+1}.pth")
    print('Model checkpoint with history saved')

if __name__ == '__main__':
    main()