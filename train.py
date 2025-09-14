import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from models.resnet18 import ResNetSentinel
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.
    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            # applying sigmoid for multi-label
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(int)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    avg_loss = running_loss / len(dataloader.dataset)

    return avg_loss, f1, precision, recall, acc

# Main Training loop
def main():
    # path to H5 datasets
    train_path = 'train_data.h5'
    val_path = 'val_data.h5'
    test_path = 'test_data.h5'

    batch_size = 16
    num_epochs = 30 # Total number of epochs
    head_epochs = 5 # Number of epochs to train only the head
    lr_head = 0.001 # Learning rate for head training
    lr_finetune = 0.0001 # Learning rate for fine-tuning

    # Resuming from checkpoint
    resume = False
    checkpoint_path = 'checkpoints/resnet_best.pth'

    # Early stopping parameters
    early_stop = True
    best_val_loss = float('inf')
    patience = 7
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

    model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=True, freeze_backbone=True).to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Value history
    train_loss_history = []
    val_loss_history = []
    val_f1_history = []
    val_prec_history = []
    val_rec_history = []
    val_acc_history = []

    if resume and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
        start_epoch = checkpoint['epoch']
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
        print("Starting head training phase")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        for epoch in range(start_epoch, head_epochs):
            print(f'\nEpoch {epoch+1}/{head_epochs}')

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, criterion, device)

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
    
    optimizer = optim.Adam(model.parameters(), lr=lr_finetune)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(max(start_epoch, head_epochs), num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, criterion, device)

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
                }, f"checkpoints/resnet_best.pth")
                print(f"Saved new best model at epoch {epoch + 1}")

            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    # Final test evaluation
    print('\nEvaluating on test set...')
    test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f}')

    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_f1': val_f1_history,
        'val_precision': val_prec_history,
        'val_recall': val_rec_history,
        'val_accuracy': val_acc_history,
    }

    # Save model checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_loss_history[-1],
        'history': history,
    }, f"checkpoints/resnet_sentinel_pretrained_epoch{epoch+1}.pth")
    print('Model checkpoint with history saved')

if __name__ == '__main__':
    main()