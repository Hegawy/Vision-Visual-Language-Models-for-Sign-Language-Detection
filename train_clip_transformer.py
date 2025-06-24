
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# --- Device ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Parameters ---
data_root = "wlasl_clip_timesformer_features_openclip"
checkpoint_dir = "wlasl_clip_transformer_checkpoints"
epochs = 10
batch_size = 4
learning_rate = 1e-4
# Auto-detect from first file
import torch
first_file = None
for root, _, files in os.walk(data_root):
    for file in files:
        if file.endswith(".pt"):
            first_file = os.path.join(root, file)
            break
    if first_file:
        break

embed_dim = torch.load(first_file).shape[1]
print(f"Detected embed_dim: {embed_dim}")

seq_len = 10

os.makedirs(checkpoint_dir, exist_ok=True)

# --- Dataset ---
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
        features = torch.load(path)  # [T, D]
        if features.shape[0] < seq_len:
            pad = torch.zeros(seq_len - features.shape[0], features.shape[1])
            features = torch.cat((features, pad), dim=0)
        elif features.shape[0] > seq_len:
            features = features[:seq_len]
        return features, label

# --- Transformer Model ---
class TemporalTransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, num_layers=2, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # [B, T, D]
        enc = self.encoder(x)  # [B, T, D]
        pooled = enc.mean(dim=1)  # [B, D]
        return self.classifier(pooled)

# --- Load Data ---
dataset = ClipFeatureSequenceDataset(data_root)
num_classes = len(dataset.class_to_idx)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# Share the same class_to_idx mapping
full_dataset = ClipFeatureSequenceDataset(data_root)
full_dataset.class_to_idx = full_dataset.class_to_idx  # enforce consistent mapping

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Model, Optimizer, Loss ---
model = TemporalTransformerClassifier(embed_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Train ---
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    total, correct, train_loss = 0, 0, 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={acc:.2f}%")

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    val_acc = 100 * correct / total
    print(f"Validation Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_transformer.pt"))
        print("✅ Saved new best model")

print("Training complete. Best validation accuracy:", best_acc)
