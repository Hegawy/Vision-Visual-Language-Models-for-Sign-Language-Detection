import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

# ---------- End-to-End CLIP + Transformer ----------
class EndToEndCLIPTransformer(nn.Module):
    def __init__(self, clip_model, num_heads=4, ff_dim=1024, num_layers=2, num_classes=7, dropout=0.1):
        super().__init__()
        self.clip = clip_model.visual
        self.embed_dim = 768  # ViT-B/32 visual encoder output

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        clip_dtype = self.clip.conv1.weight.dtype
        transformer_dtype = torch.float32
        x = x.to(dtype=clip_dtype)

        # Manually replicate CLIP ViT-B/32 forward pass up to CLS token (pre-projection)
        with torch.no_grad():
            x = self.clip.conv1(x)  # shape: [B*T, 768, 7, 7]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B*T, 768, 49]
            x = x.permute(0, 2, 1)  # [B*T, 49, 768]

            cls_token = self.clip.class_embedding.to(x.dtype)  # [1, 768]
            cls_tokens = cls_token.unsqueeze(0).expand(x.shape[0], 1, -1)  # [B*T, 1, 768]
            x = torch.cat([cls_tokens, x], dim=1)  # [B*T, 50, 768]

            x = x + self.clip.positional_embedding.to(x.dtype)
            x = self.clip.ln_pre(x)
            x = self.clip.transformer(x)
            x = self.clip.ln_post(x[:, 0, :])  # extract CLS token (shape: [B*T, 768])

        x = x.to(dtype=transformer_dtype)
        feats = x.view(B, T, -1)  # [B, T, 768]
        encoded = self.encoder(feats)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)



# ---------- Dataset ----------
class MSASLEndToEndDataset(Dataset):
    def __init__(self, root_dir, preprocess, max_frames=16):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.max_frames = max_frames
        self.samples = []
        self.class_to_idx = {}

        for idx, cls in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, cls)
            if not os.path.isdir(class_path): continue
            self.class_to_idx[cls] = idx
            for sample in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample)
                if os.path.isdir(sample_path):
                    self.samples.append((sample_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])[:self.max_frames]
        images = []

        for fname in frame_files:
            img = Image.open(os.path.join(video_path, fname)).convert("RGB")
            img = self.preprocess(img)
            images.append(img)

        while len(images) < self.max_frames:
            images.append(torch.zeros_like(images[0]))

        return torch.stack(images), label

# ---------- Training Loop ----------
def train(model, train_loader, val_loader, epochs=10, lr=1e-5):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")

# ---------- Main ----------
if __name__ == "__main__":
    train_path = "custom_MSASL/train"
    val_path = "custom_MSASL/val"

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    print("CLIP Visual Output Dim:", clip_model.visual.proj.shape[1])  # for verification

    dataset_train = MSASLEndToEndDataset(train_path, preprocess)
    dataset_val = MSASLEndToEndDataset(val_path, preprocess)

    print("✅ Detected Classes:", dataset_train.class_to_idx)

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_val, batch_size=4)

    model = EndToEndCLIPTransformer(clip_model, num_classes=len(dataset_train.class_to_idx))
    train(model, train_loader, val_loader)
