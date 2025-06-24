import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import transforms
from collections import Counter

# ---------- Frozen CLIP + Average Pooling + Classifier ----------
class FrozenCLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=7):
        super().__init__()
        self.clip = clip_model.visual.to(torch.float32)
        self.embed_dim = 768

        for param in self.clip.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        ).to(torch.float32)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W).to(torch.float32)

        x = self.clip.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = self.clip.class_embedding.to(x.dtype)
        cls_tokens = cls_token.unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.clip.positional_embedding.to(x.dtype)
        x = self.clip.ln_pre(x)
        x = self.clip.transformer(x)
        x = self.clip.ln_post(x[:, 0, :])

        x = x.view(B, T, -1)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# ---------- Dataset ----------
class MSASLEndToEndDataset(Dataset):
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
            class_samples = []
            for sample in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample)
                if os.path.isdir(sample_path):
                    class_samples.append((sample_path, idx))
            class_samples = class_samples[:15]
            self.samples.extend(class_samples)
            self.labels.extend([idx] * len(class_samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])

        if len(frame_files) > 4:
            frame_files = frame_files[2:-2]
        frame_files = frame_files[:self.max_frames]

        images = []
        for fname in frame_files:
            img = Image.open(os.path.join(video_path, fname)).convert("RGB")
            img = self.preprocess(img)
            images.append(img)

        while len(images) < self.max_frames:
            images.append(torch.zeros_like(images[0]))

        return torch.stack(images), label

# ---------- Training Loop ----------
def train(model, train_loader, val_loader, class_weights, class_names, epochs=10, lr=1e-4):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if epoch == 0:
                print(f"Example logits: {out[0].detach().cpu().numpy()}")

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

        model.eval()
        correct = total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        print(f"Validation Accuracy: {correct / total:.4f}")
        if epoch == epochs - 1:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    train_path = "custom_MSASL/train"
    val_path = "custom_MSASL/val"

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    dataset_train = MSASLEndToEndDataset(train_path, preprocess)
    dataset_val = MSASLEndToEndDataset(val_path, preprocess)

    print("✅ Detected Classes:", dataset_train.class_to_idx)
    print("Sample count:", Counter(dataset_train.labels))

    class_names = list(dataset_train.class_to_idx.keys())

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=4, num_workers=0)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(dataset_train.class_to_idx.values())),
        y=dataset_train.labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = FrozenCLIPClassifier(clip_model, num_classes=len(dataset_train.class_to_idx))
    train(model, train_loader, val_loader, class_weights_tensor, class_names)
