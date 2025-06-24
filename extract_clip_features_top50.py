import torch
import open_clip
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔌 Using device: {DEVICE}")

# Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(DEVICE).eval()

# Paths
FRAME_DIR = "PHOENIX-2014-T/features/fullFrame-210x260px/train_top50"
ANNOTATIONS = "subset_annotations_top50.csv"
SAVE_DIR = "PHOENIX-2014-T/features_clip_sla_top50"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load annotations
df = pd.read_csv(ANNOTATIONS)
label_map = {s: i for i, s in enumerate(sorted(df["orth"].unique()))}

# Feature extractor
def extract_video_features(video_path):
    frames = sorted(os.listdir(video_path))
    clip_features = []

    for frame_file in frames:
        img_path = os.path.join(video_path, frame_file)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_image(image)
            clip_features.append(feat.squeeze(0).cpu())
        except:
            continue

    return torch.stack(clip_features) if clip_features else None

# Process each unique video
for name in tqdm(df["name"].unique(), desc="Extracting CLIP features"):
    video_path = os.path.join(FRAME_DIR, name)
    features = extract_video_features(video_path)
    if features is not None:
        row = df[df["name"] == name].iloc[0]
        label = label_map[row["orth"]]
        text = row["orth"]  # Or row["translation"] if you prefer spoken sentences
        save_path = os.path.join(SAVE_DIR, f"{name}.pt")
        torch.save({"features": features, "label": label, "text": text}, save_path)

print(f"✅ Saved {len(os.listdir(SAVE_DIR))} feature files in: {SAVE_DIR}")
