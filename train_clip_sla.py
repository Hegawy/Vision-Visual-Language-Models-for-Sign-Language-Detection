import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import random
from jiwer import wer

# === CONFIG ===
FEATURE_DIR = "PHOENIX-2014-T/features_clip_sla"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 512  # or 768 depending on CLIP variant used

# === DATASET ===
class ClipFeatureDataset(Dataset):
    def __init__(self, feature_dir, metadata_file):
        self.feature_dir = feature_dir
        self.df = pd.read_csv(metadata_file)
        self.samples = self.df['name'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        label = self.labels[idx]
        path = os.path.join(self.feature_dir, f"{name}.pt")
        data = torch.load(path)
        x = data['clip_features']  # [T, D]
        return x, label

# === MODEL ===
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # [B, T, D]
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pool(x).squeeze(-1)  # [B, D]
        return self.classifier(x)

# === LOAD DATA ===
full_dataset = ClipFeatureDataset(FEATURE_DIR, os.path.join(FEATURE_DIR, "metadata.csv"))
num_classes = len(pd.read_csv(os.path.join(FEATURE_DIR, "metadata.csv"))['label'].unique())

# Split into 80/20 train/val
indices = list(range(len(full_dataset)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_loader = DataLoader(torch.utils.data.Subset(full_dataset, indices[:split]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(full_dataset, indices[split:]), batch_size=BATCH_SIZE)

# Load label index to sentence map
label_to_text = {}
with open(os.path.join(FEATURE_DIR, "label_map.txt"), "r") as f:
    for line in f:
        idx, text = line.strip().split("\t", 1)
        label_to_text[int(idx)] = text

# === MODEL SETUP ===
model = TransformerClassifier(input_dim=D_MODEL, num_classes=num_classes).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === TRAIN LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x.to(DEVICE)
        y = torch.tensor(y).to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

    train_acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # === VALIDATION + WER ===
    model.eval()
    correct = 0
    pred_texts = []
    ref_texts = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = torch.tensor(y).to(DEVICE)
            logits = model(x)
            preds = logits.argmax(1)

            for pred, true in zip(preds.cpu().tolist(), y.cpu().tolist()):
                pred_texts.append(label_to_text[pred])
                ref_texts.append(label_to_text[true])

            correct += (preds == y).sum().item()

    val_acc = correct / len(val_loader.dataset)
    val_wer = wer(ref_texts, pred_texts)
    print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}, WER: {val_wer:.4f}")
