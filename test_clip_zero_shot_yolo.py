# test_clip_zero_shot_yolo.py

import clip
import torch
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

# Load YOLOv8 for hand/upper body detection (use v8n for speed)
yolo_model = YOLO("yolov8n.pt")  # or use a custom model fine-tuned on hands

def crop_with_yolo(image_path):
    image = Image.open(image_path).convert("RGB")
    results = yolo_model.predict(image, conf=0.25, classes=[0], verbose=False)  # class 0 = person

    boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
    if len(boxes) == 0:
        print("No person detected — using full image.")
        return image

    # Use largest box
    x1, y1, x2, y2 = boxes[0]
    crop = image.crop((x1, y1, x2, y2))
    return crop

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Input image
image_path = "custom_MSASL/train/yes/yes5198/frame_005.jpg"
cropped = crop_with_yolo(image_path)
preprocessed = preprocess(cropped).unsqueeze(0).to(device)

# Define text labels
labels = ["again", "help", "no", "understand", "want", "yes", "you"]
text = clip.tokenize(labels).to(device)

# CLIP zero-shot
with torch.no_grad():
    image_features = model.encode_image(preprocessed)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1).cpu().numpy()

# Results
print("Top Prediction:", labels[probs.argmax()])
print("Confidence:", probs.max())
print("\nAll Probabilities:")
for label, prob in zip(labels, probs[0]):
    print(f"{label:>12}: {prob:.4f}")
