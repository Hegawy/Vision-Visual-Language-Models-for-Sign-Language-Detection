# train_clip_vs_resnet.py (MPS-safe version)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from PIL import Image
import numpy as np
from compare_clip_resnet import HybridVisionEncoder

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, preprocess, max_frames=16):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.max_frames = max_frames
        self.samples = []
        self.class_to_idx = {}
        self.labels = []

        for idx, cls in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, cls)
            if not os.path.isdir(class_path): continue
            self.class_to_idx[cls] = idx
            for sample in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample)
                if os.path.isdir(sample_path):
                    self.samples.append((sample_path, idx))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        if len(frame_files) > 4:
            frame_files = frame_files[2:-2]  # exclude idle frames
        frame_files = frame_files[:self.max_frames]

        frames = []
        for f in frame_files:
            img = Image.open(os.path.join(video_path, f)).convert("RGB")
            img = self.preprocess(img)
            frames.append(img)

        while len(frames) < self.max_frames:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames), label

def train_model(model, train_loader, val_loader, class_weights, class_names, epochs=10, lr=1e-4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            y = y.to(out.device)  # Ensure target matches output device
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total
        train_accs.append(train_acc)

        model.eval()
        correct = total = 0
        preds, labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    train_dir = "custom_MSASL/train"
    val_dir = "custom_MSASL/val"
    train_dataset = VideoFrameDataset(train_dir, preprocess)
    val_dataset = VideoFrameDataset(val_dir, preprocess)
    print("✅ Detected Classes:", train_dataset.class_to_idx)
    print("Sample count:", Counter(train_dataset.labels))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    class_names = list(train_dataset.class_to_idx.keys())
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(class_names)), y=train_dataset.labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # CLIP run
    print("\n===== TRAINING WITH CLIP VISUAL ENCODER =====")
    model_clip = HybridVisionEncoder(use_clip=True, num_classes=len(class_names))
    train_model(model_clip, train_loader, val_loader, class_weights_tensor, class_names)

    # ResNet run
    print("\n===== TRAINING WITH RESNET18 VISUAL ENCODER =====")
    model_resnet = HybridVisionEncoder(use_clip=False, num_classes=len(class_names))
    train_model(model_resnet, train_loader, val_loader, class_weights_tensor, class_names)
