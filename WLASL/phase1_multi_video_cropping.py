import os
import cv2
import numpy as np
from tqdm import tqdm
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from collections import defaultdict
from ultralytics import YOLO

# --- SETTINGS ---
save_root = "wlasl_phase1_multi_video_crops_upper"
os.makedirs(save_root, exist_ok=True)
max_classes = 30
min_samples_per_class = 5
frames_per_video = 10
model = YOLO("yolov8n.pt")

# --- 1. Load WLASL from Hugging Face ---
print("🔽 Downloading WLASL from Hugging Face via FiftyOne...")
dataset = fouh.load_from_hub("Voxel51/WLASL", name="wlasl_multi", max_samples=1000)
print("✅ Loaded", len(dataset), "samples")

# --- 2. Group videos by gloss ---
label_to_samples = defaultdict(list)
for sample in dataset:
    if hasattr(sample, "gloss") and sample.gloss is not None:
        label = sample.gloss.label
        label_to_samples[label].append(sample)

# --- 3. Filter classes with enough samples ---
filtered = {label: vids for label, vids in label_to_samples.items() if len(vids) >= min_samples_per_class}
selected_labels = list(filtered.keys())[:max_classes]
print(f"✅ Selected {len(selected_labels)} classes with ≥ {min_samples_per_class} videos")

# --- 4. Process each video ---
for label in tqdm(selected_labels, desc="Processing selected classes"):
    label_dir = os.path.join(save_root, label)
    os.makedirs(label_dir, exist_ok=True)
    for sample in filtered[label]:
        video_path = sample.filepath
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_per_video or total_frames <= 0:
            cap.release()
            continue

        indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
        current_index = 0
        frame_id = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if i == indices[current_index]:
                results = model(frame, verbose=False)[0]
                for box in results.boxes:
                    if int(box.cls.item()) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        h = y2 - y1
                        y1 = max(0, y1 - int(0.15 * h))
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            out_path = os.path.join(label_dir, f"{video_id}_f{frame_id}.jpg")
                            cv2.imwrite(out_path, cropped)
                            frame_id += 1
                        break
                current_index += 1
                if current_index >= len(indices):
                    break
        cap.release()

print("✅ Done: Saved cropped frames for all selected classes.")
