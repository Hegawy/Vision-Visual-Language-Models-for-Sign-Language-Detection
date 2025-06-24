# fine_tune_static_signs.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import open_clip

# --- Settings ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

batch_size = 64
epochs = 10
learning_rate = 1e-3
num_classes = 26  # EMNIST Letters A-Z

# --- Load CLIP ---
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# --- Prepare Dataset ---
transform = transforms.Compose([
    transforms.Grayscale(3),  # EMNIST is grayscale, CLIP expects 3 channels
    preprocess
])

train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Freeze CLIP image encoder ---
for param in model.visual.parameters():
    param.requires_grad = False

# --- Add Classifier Head ---
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

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.fc.parameters(), lr=learning_rate)

# --- Training Loop ---
for epoch in range(epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), (labels - 1).to(device)  # EMNIST labels are 1-indexed
        
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {100.*correct/total:.2f}%")

print("Finished Training ✅")

# --- (Optional) Save Model ---
torch.save(classifier.state_dict(), 'fine_tuned_sign_classifier.pth')
