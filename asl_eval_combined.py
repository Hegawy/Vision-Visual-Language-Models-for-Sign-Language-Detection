import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import open_clip

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Dataset Path ---
data_dir = '/Users/hegawy/Desktop/Final Project Bachelor/Synthetic ASL/Test_Alphabet'  # <-- Update this to your actual test folder

# --- Transform (match training input format) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])

# --- Dataset Loader ---
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# --- Load CLIP Model and Classifier ---
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

class SignClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x)
        return self.fc(x)

# --- Load Checkpoint ---
checkpoint = torch.load('asl_clip_finetuned_combined.pth', map_location=device)
class_names = checkpoint['class_names']
model = SignClassifier(clip_model, len(class_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Evaluation Loop ---
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# --- Filter out DEL, SPACE, NOTHING ---
excluded_labels = {'del', 'space', 'nothing'}
valid_indices = [i for i, name in enumerate(class_names) if name.lower() not in excluded_labels]
valid_class_names = [class_names[i] for i in valid_indices]

# --- Filter predictions and labels (together) ---
filtered_labels = []
filtered_preds = []

for true_label, pred_label in zip(all_labels, all_preds):
    if true_label in valid_indices:
        filtered_labels.append(true_label)
        filtered_preds.append(pred_label)

# --- Build index map and filter both predictions and labels safely ---
index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
mapped_labels = []
mapped_preds = []

for true_label, pred_label in zip(all_labels, all_preds):
    if true_label in index_map and pred_label in index_map:
        mapped_labels.append(index_map[true_label])
        mapped_preds.append(index_map[pred_label])



# --- Classification Report ---
print("\n📊 Classification Report (Filtered):\n")
print(classification_report(
    mapped_labels,
    mapped_preds,
    target_names=valid_class_names,
    zero_division=0
))

# --- Confusion Matrix ---
cm = confusion_matrix(mapped_labels, mapped_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=valid_class_names, yticklabels=valid_class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Filtered Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
