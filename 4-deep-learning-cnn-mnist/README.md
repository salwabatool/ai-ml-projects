# 🧠 Deep Learning CNN — MNIST & Fashion-MNIST

Advanced CNN classifier with residual connections, early stopping, label smoothing, and t-SNE visualisation.

## Features
- ✅ Two architectures: **BasicCNN** and **Mini ResNet** (with residual blocks)
- ✅ Data augmentation (random rotation, affine transforms)
- ✅ Label smoothing + AdamW optimiser
- ✅ ReduceLROnPlateau scheduler
- ✅ Early stopping with patience
- ✅ Confusion matrix + classification report
- ✅ **t-SNE embedding visualisation** of learned features
- ✅ Works on both MNIST and Fashion-MNIST

## Architecture

```
BasicCNN:
Input (1×28×28)
  │
  ├─ Conv Block 1: 32 filters, BN, ReLU, MaxPool, Dropout
  ├─ Conv Block 2: 64 filters, BN, ReLU, MaxPool, Dropout
  └─ FC: 512 → 128 → 10

Mini ResNet:
Input (1×28×28)
  │
  ├─ Stem: Conv 64
  ├─ Layer1: 2× ResBlock(64) → MaxPool
  ├─ Layer2: Conv 128 + ResBlock(128) → MaxPool
  └─ AdaptiveAvgPool → FC 64 → FC 10
```

## Quick Start

```bash
pip install -r requirements.txt

# Train BasicCNN on MNIST
python cnn_trainer.py --dataset mnist --arch basic --epochs 15

# Train Mini ResNet on Fashion-MNIST
python cnn_trainer.py --dataset fashion --arch resnet --epochs 20

# Outputs:
#   best_model.pth          ← best checkpoint
#   training_history.png    ← loss & accuracy curves
#   confusion_matrix.png    ← per-class confusion
#   tsne_embeddings.png     ← feature space visualisation
```

## Results (typical)

| Model | Dataset | Test Accuracy |
|-------|---------|--------------|
| BasicCNN | MNIST | ~99.2% |
| Mini ResNet | MNIST | ~99.5% |
| BasicCNN | Fashion-MNIST | ~92.3% |
| Mini ResNet | Fashion-MNIST | ~93.1% |

## 📁 Project Structure
```
4-deep-learning-cnn-mnist/
├── cnn_trainer.py    # Architectures + training loop
├── requirements.txt
└── README.md
```

## 📄 License
MIT
