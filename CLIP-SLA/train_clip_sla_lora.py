import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from jiwer import wer
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig, TaskType

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔌 Using device: {DEVICE}")
DATA_DIR = "PHOENIX-2014-T/features_clip_sla_top50"
MAX_LEN = 64
EMBED_DIM = 512
BATCH_SIZE = 16
EPOCHS = 20

# --- Load label map ---
label_map_path = os.path.join(DATA_DIR, "label_map.txt")
id2label = []
with open(label_map_path, "r") as f:
    for line in f:
        id2label.append(line.strip())
label2id = {label: idx for idx, label in enumerate(id2label)}


# --- Padding utility ---
def pad_or_truncate(tensor, max_len=MAX_LEN):
    length = tensor.shape[0]
    if length > max_len:
        return tensor[:max_len]
    elif length < max_len:
        pad = torch.zeros(max_len - length, EMBED_DIM)
        return torch.cat([tensor, pad], dim=0)
    return tensor

# --- Dataset ---
class CLIPFeatureDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for file in os.listdir(data_dir):
            if file.endswith(".pt"):
                sample = torch.load(os.path.join(data_dir, file))
                feat = pad_or_truncate(sample["features"])
                label = label2id[sample["text"]]
                text = sample["text"]
                self.samples.append((feat, label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- Transformer with LoRA ---
class TransformerWithLoRA(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_classes=50, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_LEN, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# --- Prepare dataset and dataloaders ---
dataset = CLIPFeatureDataset(DATA_DIR)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# --- Wrap model with LoRA ---
base_model = TransformerWithLoRA(num_classes=len(id2label)).to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Safe default, update if needed
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
model = get_peft_model(base_model, lora_config)

# --- Training setup ---
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct = 0, 0
    for feats, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_set)
    print(f"Epoch {epoch}: Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # --- Validation + WER ---
    model.eval()
    all_preds, all_labels, pred_texts, gt_texts = [], [], [], []

    with torch.no_grad():
        for feats, labels, texts in val_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pred_texts.extend([id2label[p] for p in preds.cpu().tolist()])
            gt_texts.extend(texts)

    val_acc = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_labels)
    val_wer = wer(gt_texts, pred_texts)
    print(f"Epoch {epoch}: Val Acc: {val_acc:.4f}, WER: {val_wer:.4f}")
