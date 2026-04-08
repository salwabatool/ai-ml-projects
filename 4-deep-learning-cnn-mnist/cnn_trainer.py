"""
Deep Learning — CNN on MNIST / Fashion-MNIST
Covers: CNN architecture, training loop, LR scheduling,
        early stopping, confusion matrix, t-SNE embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import argparse
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MNIST_CLASSES      = [str(i) for i in range(10)]
FASHION_CLASSES    = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ─────────────────────────────────────────────
# 1. ARCHITECTURES
# ─────────────────────────────────────────────
class BasicCNN(nn.Module):
    """Simple 2-conv-block CNN."""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv2(self.conv1(x)))

    def extract_features(self, x):
        """Return penultimate-layer embeddings for t-SNE."""
        x = self.conv2(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc[1](self.fc[0](x))   # Flatten + Linear1 + ReLU
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class ResNet_Mini(nn.Module):
    """Tiny ResNet for MNIST."""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResidualBlock(128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.layer2(self.layer1(self.stem(x))))


# ─────────────────────────────────────────────
# 2. DATA
# ─────────────────────────────────────────────
def get_loaders(dataset="mnist", batch_size=128, data_dir="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    aug_tf = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    DS = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
    train_ds = DS(data_dir, train=True, download=True, transform=aug_tf)
    test_ds  = DS(data_dir, train=False, download=True, transform=tf)
    n_val = int(0.1 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])
    return (
        DataLoader(train_ds, batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(val_ds,   batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds,  batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


# ─────────────────────────────────────────────
# 3. EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best = None; self.stop = False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    es = EarlyStopping(patience=7)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step(vl_loss); es(vl_loss)

        for k, v in zip(history, [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        print(f"  Epoch {epoch:3d} | TrLoss={tr_loss:.4f} TrAcc={tr_acc:.2f}%"
              f" | VlLoss={vl_loss:.4f} VlAcc={vl_acc:.2f}%")

        if vl_acc == max(history["val_acc"]):
            torch.save(model.state_dict(), "best_model.pth")

        if es.stop:
            print(f"  ⏹  Early stopping at epoch {epoch}")
            break

    return history


# ─────────────────────────────────────────────
# 5. EVALUATION & VISUALS
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes,
            [("train_loss","val_loss"), ("train_acc","val_acc")],
            ["Loss", "Accuracy"]):
        for k in key:
            ax.plot(history[k], label=k)
        ax.set_title(title); ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig("training_history.png", dpi=120); plt.close()
    print("  → training_history.png")


@torch.no_grad()
def full_evaluation(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu()
        all_preds.extend(preds.numpy()); all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    acc = 100.0 * (all_preds == all_labels).mean()
    print(f"\n  Test Accuracy: {acc:.2f}%")
    print("\n" + classification_report(all_labels, all_preds,
                                        target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120); plt.close()
    print("  → confusion_matrix.png")
    return acc


@torch.no_grad()
def tsne_embeddings(model, test_loader, class_names, n_samples=2000):
    if not isinstance(model, BasicCNN):
        return
    model.eval()
    feats, labels = [], []
    for imgs, lbl in test_loader:
        feats.append(model.extract_features(imgs.to(DEVICE)).cpu().numpy())
        labels.extend(lbl.numpy())
        if len(labels) >= n_samples:
            break
    feats = np.concatenate(feats)[:n_samples]
    labels = np.array(labels[:n_samples])

    print("  Computing t-SNE …")
    emb = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(feats)

    plt.figure(figsize=(10, 8))
    palette = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, cls in enumerate(class_names):
        mask = labels == i
        plt.scatter(emb[mask, 0], emb[mask, 1], c=[palette[i]],
                    s=5, alpha=0.6, label=cls)
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1))
    plt.title("t-SNE Embeddings"); plt.tight_layout()
    plt.savefig("tsne_embeddings.png", dpi=120); plt.close()
    print("  → tsne_embeddings.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--arch",     choices=["basic", "resnet"],  default="basic")
    parser.add_argument("--epochs",   type=int,   default=15)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=128)
    args = parser.parse_args()

    class_names = MNIST_CLASSES if args.dataset == "mnist" else FASHION_CLASSES

    print(f"\n{'='*60}")
    print(f"  Deep Learning CNN  |  Dataset={args.dataset}  Arch={args.arch}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_loader = get_loaders(args.dataset, args.batch)

    model = (BasicCNN() if args.arch == "basic" else ResNet_Mini()).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}\n")

    history = train_model(model, train_loader, val_loader, args.epochs, args.lr)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

    plot_history(history)
    full_evaluation(model, test_loader, class_names)
    tsne_embeddings(model, test_loader, class_names)

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
