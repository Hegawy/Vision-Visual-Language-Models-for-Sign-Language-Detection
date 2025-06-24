import os
import cv2
import numpy as np
from tqdm import tqdm
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from ultralytics import YOLO

# --- SETTINGS ---
save_root = "wlasl_phase1_crops_upper"
os.makedirs(save_root, exist_ok=True)
num_words = 100
frames_per_video = 10
model = YOLO("yolov8n.pt")  # lightweight and fast

# --- 1. Load WLASL subset from Hugging Face ---
print("🔽 Downloading WLASL from Hugging Face via FiftyOne...")
dataset = fouh.load_from_hub("Voxel51/WLASL", name="wlasl_phase1", max_samples=500)
print("✅ Loaded", len(dataset), "samples")

# --- 2. Filter for ~100 unique words ---
gloss_labels = set()
filtered_samples = []

for sample in dataset:
    if hasattr(sample, "gloss") and sample.gloss is not None:
        label = sample.gloss.label
        if label not in gloss_labels:
            gloss_labels.add(label)
            filtered_samples.append(sample)
    if len(gloss_labels) >= num_words:
        break


print(f"✅ Selected {len(gloss_labels)} unique signs")

# --- 3. Extract + crop frames ---
for sample in tqdm(filtered_samples, desc="Processing videos"):
    video_path = sample.filepath
    label = sample.gloss.label 
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    label_dir = os.path.join(save_root, label)
    os.makedirs(label_dir, exist_ok=True)

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
                if int(box.cls.item()) == 0:  # 'person' class
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h = y2 - y1
                    y1 = max(0, y1 - int(0.15 * h))  # pad for head/shoulders
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        out_path = os.path.join(label_dir, f"{video_id}_f{frame_id}.jpg")
                        cv2.imwrite(out_path, cropped)
                        frame_id += 1
                    break  # only first person box
            current_index += 1
            if current_index >= len(indices):
                break

    cap.release()

print("✅ Done: Cropped upper body frames saved in", save_root)
