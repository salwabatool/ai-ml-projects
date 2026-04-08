"""
Computer Vision Image Classifier
- Transfer Learning with ResNet-50 (fine-tuned on CIFAR-10)
- Custom CNN from scratch for comparison
- Grad-CAM visualisation for explainability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import time

# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────
# 1. CUSTOM CNN (built from scratch)
# ─────────────────────────────────────────────
class CustomCNN(nn.Module):
    """Lightweight CNN: 3 conv blocks + 2 FC layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 2. TRANSFER LEARNING — ResNet-50
# ─────────────────────────────────────────────
def build_resnet50(num_classes: int = 10, freeze_backbone: bool = True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────
# 3. DATA LOADERS
# ─────────────────────────────────────────────
def get_loaders(batch_size: int = 64, data_dir: str = "./data"):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,
                                            download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False,
                                            download=True, transform=test_tf)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True,
                                    num_workers=2, pin_memory=True),
        torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True),
    )


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion, epoch: int):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(loader):.4f} | Train Acc: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"             Val Acc : {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────
# 5. GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ─────────────────────────────────────────────
# 6. VISUALISATION HELPERS
# ─────────────────────────────────────────────
def plot_training_history(train_accs, val_accs, save_path="training_history.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy", marker="o")
    plt.plot(val_accs,   label="Val Accuracy",   marker="s")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Training History"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved training history → {save_path}")


def show_predictions(model, loader, n=8):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        preds = model(imgs).argmax(1).cpu()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    for i, ax in enumerate(axes.flat):
        img = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)
        ax.imshow(img)
        colour = "green" if preds[i] == labels[i] else "red"
        ax.set_title(
            f"P:{CIFAR10_CLASSES[preds[i]]}\nT:{CIFAR10_CLASSES[labels[i]]}",
            color=colour, fontsize=9,
        )
        ax.axis("off")
    plt.suptitle("Predictions (green=correct, red=wrong)")
    plt.tight_layout()
    plt.savefig("predictions.png"); plt.close()
    print("  Saved predictions → predictions.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["cnn", "resnet50"], default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--batch",  type=int, default=64)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Computer Vision Classifier  |  device={DEVICE}")
    print(f"  Model: {args.model}  |  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    train_loader, test_loader = get_loaders(args.batch)

    if args.model == "resnet50":
        model = build_resnet50().to(DEVICE)
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    else:
        model = CustomCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    train_accs, val_accs = [], []
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        t_acc = train(model, train_loader, optimizer, criterion, epoch)
        v_acc = evaluate(model, test_loader)
        scheduler.step()
        train_accs.append(t_acc); val_accs.append(v_acc)
        if v_acc > best_val:
            best_val = v_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\n  Best Val Accuracy: {best_val:.2f}%")
    plot_training_history(train_accs, val_accs)
    show_predictions(model, test_loader)
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
