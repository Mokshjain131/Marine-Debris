import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from models.resnet18 import ResNetSentinel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    num_epochs = 100
    lr = 0.001

    # Resuming from checkpoint
    resume = True
    checkpoint_path = 'checkpoints/resnet_best_25.pth'

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    counter = 0

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
    model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=False).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

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

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'val_f1_history': val_f1_history,
                'val_prec_history': val_prec_history,
                'val_rec_history': val_rec_history,
                'val_acc_history': val_acc_history,
                'best_val_loss': best_val_loss,
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
    }, "checkpoints/resnet_sentinel_epoch100.pth")
    print('Model checkpoint with history saved')

    # Plotting metrics history against iterations
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 8))

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    # F1 Score
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_f1_history, label='Val F1', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over epochs')
    plt.legend()

    # Precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_prec_history, label='Val Precision', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision over epochs')
    plt.legend()

    # Recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_rec_history, label='Val Recall', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over epochs')
    plt.legend()

    # Accuracy
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_acc_history, label='Val Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save plots
    os.makedirs("checkpoints", exist_ok=True)
    plt.savefig("checkpoints/training_curves_100.png")
    print("Training curves saved as png")

if __name__ == '__main__':
    main()