import clip
import torch
from PIL import Image

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Replace this path with the path to one of your actual sign frames
image_path = "custom_MSASL/train/yes/yes5198/frame_005.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Define the label set
labels = ["again", "help", "no", "understand", "want", "yes", "you"]
text = clip.tokenize(labels).to(device)

# Perform zero-shot classification
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1).cpu().numpy()

# Output results
print("Top Prediction:", labels[probs.argmax()])
print("Confidence:", probs.max())
print("\nAll Probabilities:")
for label, prob in zip(labels, probs[0]):
    print(f"{label:>12}: {prob:.4f}")
