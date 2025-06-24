import os
import cv2
import numpy as np
from tqdm import tqdm
import fiftyone as fo

# Load dataset
dataset = fo.load_dataset("wlasl")

# Output directory
dataset_dir = "./wlasl_subset_frames"
os.makedirs(dataset_dir, exist_ok=True)

# How many frames to extract per video
num_frames = 10

for sample in tqdm(dataset):
    video_path = sample.filepath
    label = sample.gloss.label
    label_folder = os.path.join(dataset_dir, label)
    os.makedirs(label_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames))  # take all if fewer than num_frames
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frame_id = 0
    current_index = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i == frame_indices[current_index]:
            frame_filename = os.path.join(label_folder, f"{sample.id}_{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
            current_index += 1
            if current_index >= len(frame_indices):
                break

    cap.release()
