
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# --- Paths ---
data_root = "wlasl_phase1_top30_upper"
checkpoint_path = "wlasl_clip_finetune_top30_checkpoints/best_clip_finetuned.pt"
batch_size = 4
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# --- Load CLIP ---
import open_clip
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)
clip_model.visual.requires_grad_(True)

# --- Fine-tuned model wrapper ---
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

# --- Dataset ---
val_transform = preprocess
dataset = datasets.ImageFolder(data_root, transform=val_transform)
class_names = dataset.classes
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Load Model ---
model = CLIPFineTuner(clip_model, len(class_names)).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Evaluate ---
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Report ---
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
