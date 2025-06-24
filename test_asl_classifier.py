import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import os
import open_clip

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load model ---
checkpoint = torch.load('asl_clip_finetuned_combined.pth', map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)

# Load CLIP
clip_model, _, preprocess_clip = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device).eval()

# Define classifier
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

model = SignClassifier(clip_model, num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])

# --- Predict single image ---
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        top = probs.argmax(dim=1).item()
    return class_names[top], probs[0][top].item()

# --- Predict all images in folder ---
def test_folder(folder_path):
    correct = 0
    total = 0
    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(folder_path, file)
        filename_no_ext = os.path.splitext(file)[0]
        true_label = filename_no_ext.split('_')[0].upper()
        pred, prob = predict_image(path)
        total += 1
        if pred.upper() == true_label:
            correct += 1
        print(f"🖼 {file} | True: {true_label} | Predicted: {pred} | Confidence: {prob*100:.2f}%")
    if total > 0:
        print(f"\n✅ Overall Accuracy: {100. * correct / total:.2f}% ({correct}/{total})")
    else:
        print("⚠️ No images found.")


# --- Entry point ---
if __name__ == "__main__":
    # OPTION 1: Test a single image
    image_path = "Y1.jpg"
    label, confidence = predict_image(image_path)
    print(f"\nSingle Image Prediction: {label} ({confidence*100:.2f}%)")

    # OPTION 2: Test a folder of labeled images (e.g., test set from ASL dataset)
    # test_folder("/Users/hegawy/Desktop/Final Project Bachelor/ASL alphabet Dataset/asl_alphabet_test/asl_alphabet_test")  # Replace with your folder path
