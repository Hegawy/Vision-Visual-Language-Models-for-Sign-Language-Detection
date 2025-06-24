
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import classification_report
import numpy as np

# --- Settings ---
data_root = "wlasl_clip_features_top30"
checkpoint_path = "wlasl_clip_transformer_top30_checkpoints/best_transformer.pt"
batch_size = 4
seq_len = 10
device = "mps" if torch.backends.mps.is_available() else "cpu"

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
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        features = torch.load(path)
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

    def forward(self, x):
        enc = self.encoder(x)
        pooled = enc.mean(dim=1)
        return self.classifier(pooled)

# --- Load dataset and model ---
dataset = ClipFeatureSequenceDataset(data_root)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
num_classes = len(dataset.class_to_idx)

# Detect embedding dimension
sample_tensor = torch.load(dataset.samples[0][0])
embed_dim = sample_tensor.shape[1]

model = TemporalTransformerClassifier(embed_dim, num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Evaluation ---
all_preds = []
all_labels = []

with torch.no_grad():
    for clips, labels in val_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Report ---
idx_to_class = dataset.idx_to_class
target_names = [idx_to_class[i] for i in range(num_classes)]
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))
