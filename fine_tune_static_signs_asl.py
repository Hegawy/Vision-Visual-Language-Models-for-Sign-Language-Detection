import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import open_clip

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Hyperparameters ---
batch_size = 64
epochs = 10
learning_rate = 1e-4

# --- Load CLIP ---
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# --- Dataset ---
data_dir = '/Users/hegawy/Desktop/Final Project Bachelor/ASL alphabet Dataset/asl_alphabet_train/asl_alphabet_train'  # point this to your extracted ASL Alphabet directory
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# --- Classifier ---
class SignClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x)
        logits = self.fc(x)
        return logits

classifier = SignClassifier(model, num_classes).to(device)

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.fc.parameters(), lr=learning_rate)

print("🔁 Starting Fine-Tuning...")
for epoch in range(epochs):
    classifier.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} Accuracy: {acc:.2f}%")

print("✅ Finished Fine-Tuning")

# --- Save Model ---
torch.save({
    'model_state_dict': classifier.state_dict(),
    'class_names': class_names
}, 'asl_clip_finetuned.pth')
