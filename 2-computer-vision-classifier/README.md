# 🖼️ Computer Vision Image Classifier

Transfer learning + custom CNN on **CIFAR-10** with **Grad-CAM** explainability.

## Features
- ✅ Custom CNN (3 conv-blocks, BatchNorm, Dropout)
- ✅ ResNet-50 Transfer Learning (frozen backbone + fine-tuned head)
- ✅ Grad-CAM heatmaps for model explainability
- ✅ Training history plots & prediction visualisation
- ✅ Cosine annealing LR scheduler

## Architecture

```
Input (32×32 RGB)
      │
  ┌───▼────────────────────────────────────────┐
  │  Custom CNN                                │
  │  Conv Block 1: 32 filters → MaxPool        │
  │  Conv Block 2: 64 filters → MaxPool        │
  │  Conv Block 3: 128 filters → MaxPool       │
  │  FC 512 → FC 10 → Softmax                  │
  └────────────────────────────────────────────┘
                  OR
  ┌───▼────────────────────────────────────────┐
  │  ResNet-50 (ImageNet pretrained)           │
  │  Frozen backbone + replaced FC head        │
  └────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# Train Custom CNN (10 epochs)
python classifier.py --model cnn --epochs 10

# Train ResNet-50 (transfer learning)
python classifier.py --model resnet50 --epochs 5

# Outputs:
#   best_model.pth        ← saved weights
#   training_history.png  ← accuracy curves
#   predictions.png       ← sample predictions
```

## Results (typical)

| Model | Val Accuracy | Params |
|-------|-------------|--------|
| Custom CNN | ~82% | ~1.2M |
| ResNet-50 (TL) | ~90% | ~23M |

## Dataset
**CIFAR-10** — 60,000 images (32×32), 10 classes, auto-downloaded via `torchvision`.

## 📁 Project Structure
```
2-computer-vision-classifier/
├── classifier.py     # Model, training, Grad-CAM
├── requirements.txt
└── README.md
```

## 📄 License
MIT
