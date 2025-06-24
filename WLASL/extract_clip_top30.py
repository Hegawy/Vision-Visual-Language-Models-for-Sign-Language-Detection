
import os
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

# --- Paths ---
input_dir = "wlasl_phase1_top30_upper"
output_dir = "wlasl_clip_features_top30"
os.makedirs(output_dir, exist_ok=True)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load CLIP Model ---
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# --- Traverse Labels ---
for label in tqdm(os.listdir(input_dir), desc="Processing labels"):
    label_path = os.path.join(input_dir, label)
    output_label_path = os.path.join(output_dir, label)
    os.makedirs(output_label_path, exist_ok=True)

    # Group frames by video_id prefix
    video_frames = {}
    for fname in sorted(os.listdir(label_path)):
        if not fname.endswith(('.jpg', '.png')):
            continue
        vid = "_".join(fname.split("_")[:-1])  # remove _f0, _f1...
        video_frames.setdefault(vid, []).append(os.path.join(label_path, fname))

    # Process each video
    for vid, frame_paths in video_frames.items():
        features = []

        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(image_tensor).squeeze(0).cpu()
            features.append(embedding)

        features_tensor = torch.stack(features)  # shape: [T, D]
        torch.save(features_tensor, os.path.join(output_label_path, f"{vid}.pt"))

print("✅ Done extracting CLIP features.")
