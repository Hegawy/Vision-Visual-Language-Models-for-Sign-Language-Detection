import os
import torch
import open_clip
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# Prompts for ASL letters
classes = [chr(i) for i in range(65, 91)]  # A-Z
prompts = [f"A photo of the American Sign Language letter {c}" for c in classes]
tokenized = tokenizer(prompts).to(device)

# Load one image
img_path = "/Users/hegawy/Desktop/Final Project Bachelor/ASL alphabet Dataset/asl_alphabet_train/asl_alphabet_train/B/B1.jpg"  # Replace with your image path
image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

# Encode
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(tokenized)
    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1)

# Output top prediction
top = probs[0].topk(1)
print(f"Predicted Letter: {classes[top.indices[0]]}, Confidence: {top.values[0].item()*100:.2f}%")
