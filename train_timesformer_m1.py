
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

# --- TimeSformer simplified variant ---
from timesformer.models.vit import TimeSformer

# --- Device setup for M1 Pro ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Paths ---
data_root = "wlasl_clip_timesformer_features_openclip"
checkpoint_dir = "wlasl_timesformer_checkpoints_m1"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Parameters ---
epochs = 10
batch_size = 4
learning_rate = 1e-4
embed_dim = 768
max_seq_len = 10

# --- Custom Dataset ---
class ClipFeatureSequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.class_to_idx = {}
        for idx, label in enumerate(sorted(os.listdir(root_dir))):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            self.class_to_idx[label] = idx
            for file in os.listdir(label_dir):
                if file.endswith(".pt"):
                    self.samples.append((os.path.join(label_dir, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        features = torch.load(path)  # shape: [T, D]
        if features.shape[0] < max_seq_len:
            pad = torch.zeros(max_seq_len - features.shape[0], features.shape[1])
            features = torch.cat((features, pad), dim=0)
        elif features.shape[0] > max_seq_len:
            features = features[:max_seq_len]
        return features, label

# --- Load dataset and split ---
full_dataset = ClipFeatureSequenceDataset(data_root)
num_classes = len(full_dataset.class_to_idx)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- TimeSformer Model Setup ---
class TimeSformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.timesformer = TimeSformer(img_size=224, num_classes=0, num_frames=max_seq_len,
                                       attention_type='divided_space_time',
                                       in_chans=embed_dim)  # lightweight for M1
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [B, T, D]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, D, T, 1]
        feats = self.timesformer(x)  # [B, D]
        return self.classifier(feats)

model = TimeSformerClassifier(embed_dim=embed_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training loop ---
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | Train Acc={train_acc:.2f}%")

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Acc: {val_acc:.2f}%")

    # --- Save best checkpoint ---
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_timesformer.pt"))
        print("✅ Saved new best model")

print("Training complete. Best validation accuracy:", best_acc)
