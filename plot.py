import torch
import matplotlib.pyplot as plt
import os

def plot_from_checkpoint(checkpoint_path, save_path=None):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract history (supports both old/new style)
    history = checkpoint.get('history', {
        'train_loss': checkpoint.get('train_loss_history', []),
        'val_loss': checkpoint.get('val_loss_history', []),
        'val_f1': checkpoint.get('val_f1_history', []),
        'val_precision': checkpoint.get('val_prec_history', []),
        'val_recall': checkpoint.get('val_rec_history', []),
        'val_accuracy': checkpoint.get('val_acc_history', []),
    })

    # Epochs axis
    epochs = range(1, len(history['train_loss']) + 1)
    print(epochs)

    # Create plots
    plt.figure(figsize=(14, 8))

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    # F1 Score
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['val_f1'], label='Val F1', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over epochs')
    plt.legend()

    # Precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['val_precision'], label='Val Precision', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision over epochs')
    plt.legend()

    # Recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['val_recall'], label='Val Recall', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over epochs')
    plt.legend()

    # Accuracy
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plots saved at {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    checkpoint_file = "MARIDA_models/resnet/trained_models/18/model.pth"
    plot_from_checkpoint(checkpoint_file, save_path="pretrained/pretrained.png")
