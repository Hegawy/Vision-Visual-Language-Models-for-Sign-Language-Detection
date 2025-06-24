import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import open_clip

# === SETTINGS ===
ANNOTATIONS_FILE = "PHOENIX-2014-T/annotations/manual/subsets/train_sentence_subset.csv"
FRAME_ROOT_DIR = "PHOENIX-2014-T/features/fullFrame-210x260px/train_top50"
FEATURES_OUT_DIR = "PHOENIX-2014-T/features_clip_sla"
FRAMES_PER_VIDEO = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# === STEP 1: Load annotations and create label map ===
df = pd.read_csv(ANNOTATIONS_FILE, delimiter='|', header=None)
df.columns = ['name', 'video', 'start', 'end', 'speaker', 'orth', 'translation']
df = df[['name', 'orth']]

label_map = {s: i for i, s in enumerate(df['orth'].unique())}

# === STEP 2: Load CLIP model ===
model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
model = model.to(DEVICE).eval()

# === STEP 3: Create output directory ===
os.makedirs(FEATURES_OUT_DIR, exist_ok=True)

# === STEP 4: Process each video ===
results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLIP features"):
    name = row['name']
    sentence = row['orth']
    label = label_map[sentence]
    video_dir = os.path.join(FRAME_ROOT_DIR, name)

    if not os.path.exists(video_dir):
        continue

    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
    if len(frame_files) < FRAMES_PER_VIDEO:
        continue

    indices = np.linspace(0, len(frame_files) - 1, FRAMES_PER_VIDEO).astype(int)
    images = []
    for idx in indices:
        try:
            frame = Image.open(os.path.join(video_dir, frame_files[idx])).convert("RGB")
            images.append(preprocess(frame))
        except:
            continue

    if len(images) != FRAMES_PER_VIDEO:
        continue

    batch = torch.stack(images).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(batch)  # shape: [T, 512 or 768]

    save_path = os.path.join(FEATURES_OUT_DIR, f"{name}.pt")
    torch.save({
        "clip_features": features.cpu(),
        "label": label,
        "text": sentence
    }, save_path)

    results.append((name, label, sentence))

# === Save metadata and label map ===
pd.DataFrame(results, columns=["name", "label", "text"]).to_csv(os.path.join(FEATURES_OUT_DIR, "metadata.csv"), index=False)
with open(os.path.join(FEATURES_OUT_DIR, "label_map.txt"), "w") as f:
    for text, idx in label_map.items():
        f.write(f"{idx}\t{text}\n")

print(f"✅ Saved {len(results)} feature files in: {FEATURES_OUT_DIR}")
