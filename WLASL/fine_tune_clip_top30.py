
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm

# --- Paths ---
data_root = "wlasl_phase1_top30_upper"
checkpoint_dir = "wlasl_clip_finetune_top30_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Settings ---
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
batch_size = 4
epochs = 10
learning_rate = 5e-5

# --- Load CLIP model (trainable) ---
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
model.visual.requires_grad_(True)  # Unfreeze visual encoder

# --- Classifier head ---
class CLIPFineTuner(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Sequential(
            nn.LayerNorm(clip_model.visual.output_dim),
            nn.Linear(clip_model.visual.output_dim, num_classes)
        )

    def forward(self, image):
        features = self.clip_model.encode_image(image)
        return self.fc(features)

# --- Prepare dataset ---
train_transform = preprocess


dataset = datasets.ImageFolder(data_root, transform=train_transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Model, Optimizer, Loss ---
model = CLIPFineTuner(model, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# --- Training Loop ---
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={acc:.2f}%")

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_clip_finetuned.pt"))
        print("✅ Saved new best model")

print("✅ Training complete. Best validation accuracy:", best_acc)
