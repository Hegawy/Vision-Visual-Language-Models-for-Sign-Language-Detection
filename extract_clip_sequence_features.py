
import os
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from torchvision import transforms

# --- Paths ---
frame_root = "wlasl_phase1_top30_upper"
output_root = "wlasl_clip_sequence_features_top30"
checkpoint_path = "wlasl_clip_finetune_top30_checkpoints/best_clip_finetuned.pt"
os.makedirs(output_root, exist_ok=True)

# --- Device ---
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load fine-tuned CLIP ---
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)
clip_model.visual.requires_grad_(True)

# --- Wrap in fine-tuned head ---
class CLIPFineTuner(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image):
        return self.clip_model.encode_image(image)

model = CLIPFineTuner(clip_model).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
model.eval()

# --- Process each label ---
for label in tqdm(os.listdir(frame_root), desc="Processing labels"):
    label_path = os.path.join(frame_root, label)
    output_label_path = os.path.join(output_root, label)
    os.makedirs(output_label_path, exist_ok=True)

    # Group frames by video ID
    video_frames = {}
    for fname in sorted(os.listdir(label_path)):
        if not fname.endswith(('.jpg', '.png')):
            continue
        vid = "_".join(fname.split("_")[:-1])
        video_frames.setdefault(vid, []).append(os.path.join(label_path, fname))

    # Process each video
    for vid, frame_paths in video_frames.items():
        features = []

        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image_tensor).squeeze(0).cpu()
            features.append(embedding)

        features_tensor = torch.stack(features)  # [T, 512]
        torch.save(features_tensor, os.path.join(output_label_path, f"{vid}.pt"))

print("✅ Done: CLIP sequence features saved.")
