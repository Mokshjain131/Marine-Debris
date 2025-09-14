from models.resnet18 import ResNetSentinel
import torch

model = ResNetSentinel(num_bands=11, num_classes=15, pretrained=True)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

for param in model.parameters():
    param.requires_grad = False
for param in model.model.fc.parameters():
    param.requires_grad = True

print("\nAfter freezing backbone:")
trainable_params_head = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params_head:,}")