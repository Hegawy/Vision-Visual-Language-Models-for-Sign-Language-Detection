import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

# 1. Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Prepare dataset
data_dir = "./wlasl_subset_frames"
batch_size = 32
num_epochs = 10
learning_rate = 1e-5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. Prepare label tokens
class_names = dataset.classes
text_tokens = clip.tokenize(class_names).to(device)

# 4. Fine-tuning setup
model.eval()  # CLIP normally frozen
image_encoder = model.visual
logit_scale = model.logit_scale.exp().item()

# Add classifier head on top of CLIP image embeddings
image_features_dim = image_encoder(torch.randn(1, 3, 224, 224).to(device)).shape[1]
classifier = nn.Linear(image_features_dim, len(class_names)).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# 5. Training loop
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = image_encoder(images)

        outputs = classifier(image_features)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples * 100
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

# 6. Save model
torch.save(classifier.state_dict(), "clip_wlasl_classifier.pt")
print("✅ Fine-tuning complete. Model saved.")
