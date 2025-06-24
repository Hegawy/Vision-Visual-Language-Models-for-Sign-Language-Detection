import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import open_clip
import os

# ===== 1. Device setup =====
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ===== 2. Load CLIP model =====
clip_model, _, preprocess_clip = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)
clip_model.eval()

# ===== 3. Define your classifier =====
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

# ===== 4. Load the fine-tuned model weights =====
num_classes = 26  # A-Z
classifier = SignClassifier(clip_model, num_classes).to(device)
classifier.load_state_dict(torch.load('fine_tuned_sign_classifier.pth', map_location=device))
classifier.eval()
print("Loaded fine-tuned model.")

# ===== 5. Transform for new images =====
transform = transforms.Compose([
    transforms.Grayscale(3),  # EMNIST was grayscale, CLIP expects RGB
    preprocess_clip
])

# ===== 6. Helper function to predict single image =====
def predict_image(image_path):
    image = Image.open(image_path).convert('L')  # Load as grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    outputs = classifier(image)
    
    # Print raw logits for debugging
    print("Raw Logits:", outputs)
    
    _, predicted = outputs.max(1)
    predicted_letter = chr(predicted.item() + ord('A'))  # 0 -> 'A', 1 -> 'B', etc.
    return predicted_letter


# ===== 7. Main testing =====
def test_folder(folder_path):
    correct = 0
    total = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            true_label = filename[0].upper()  # Assume filenames like 'A1.png', 'B2.jpg'
            image_path = os.path.join(folder_path, filename)
            predicted = predict_image(image_path)
            print(f"Image: {filename} | Predicted: {predicted} | True: {true_label}")

            if predicted == true_label:
                correct += 1
            total += 1
    
    if total > 0:
        print(f"\nOverall Accuracy: {100. * correct / total:.2f}% ({correct}/{total})")
    else:
        print("No images found.")

# ===== 8. Run it =====
if __name__ == "__main__":
    # Option 1: Test a single image
     img_path = "W1.jpg"
     prediction = predict_image(img_path)
     print(f"Predicted Letter: {prediction}")

    # Option 2: Test all images in a folder
    # test_folder('path_to_your_test_images')
