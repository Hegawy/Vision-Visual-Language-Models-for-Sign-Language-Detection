import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import open_clip
from open_clip import tokenize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


###### Loading the Sign Language MNIST dataset A-E only #######

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(3),  # Make 1 channel -> 3 channels (CLIP expects 3 channels)
    transforms.Resize((224, 224)),  # Resize to CLIP's input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Download dataset
train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# Limit to a small number of classes (for now A-E → 0-4)
selected_classes = [0, 1, 2, 3, 4]

def filter_dataset(dataset):
    idx = torch.isin(dataset.targets, torch.tensor(selected_classes))
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    return dataset

train_dataset = filter_dataset(train_dataset)
test_dataset = filter_dataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

##### Loading Pre-trained CLIP model ##### 

device = "mps" if torch.backends.mps.is_available() else "cpu"

model, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

#### Giving CLIP text prompts describing each letter ####

class_names = ["A", "B", "C", "D", "E"]

prompts = [f"an image of the letter {c} in sign language" for c in class_names]
tokenized_prompts = open_clip.tokenize(prompts).to(device)

# Get text embeddings
with torch.no_grad():
    text_features = model.encode_text(tokenized_prompts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

## Now CLIP knows what we expect for each letter

##### Image -> Feature extraction -> Prediction

def predict_batch(images):
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (100.0 * image_features @ text_features.T)

        # Pick the highest scored class
        predicted = similarity.argmax(dim=1)
        return predicted
    

#### Evaluation

all_preds = []
all_labels = []

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    preds = predict_batch(images)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Zero-Shot CLIP Sign Detection Accuracy: {accuracy * 100:.2f}%")

