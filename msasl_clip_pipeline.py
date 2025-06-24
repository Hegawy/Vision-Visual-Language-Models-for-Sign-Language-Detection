# msasl_clip_pipeline.py

import os
import torch
from glob import glob
from PIL import Image
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import clip
import cv2

# ==== CONFIGURATION ====
TARGET_CLASSES = ["yes", "no", "please", "understand", "help", "sorry"]
DATASET_PATHS = {
    "train": "processed_data_MS_ASL100_Train",
    "val": "processed_data_MS_ASL100_Val",
    "test": "processed_data_MS_ASL100_Test"
}
CROPPED_DIR = "MSASL_CROPPED"
MAX_FRAMES = 8

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
yolo_model = YOLO("yolov8n.pt")

# ==== LOAD DATASET ====
def load_samples_flat(split_dir, target_classes):
    data = []
    label_map = {label: idx for idx, label in enumerate(sorted(target_classes))}
    for label in target_classes:
        label_path = os.path.join(split_dir, label)
        if not os.path.exists(label_path): continue
        for sample in os.listdir(label_path):
            sample_path = os.path.join(label_path, sample)
            frame_paths = sorted(glob(os.path.join(sample_path, "frame_*.jpg")))
            if frame_paths:
                data.append((frame_paths, label_map[label]))
    return data, label_map

# ==== CROP FRAMES USING YOLO ====
def crop_upper_body(image_path, save_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)[0]
    for box in results.boxes:
        if int(box.cls) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]
            cv2.imwrite(save_path, cropped)
            return
    cv2.imwrite(save_path, img)

def crop_all_frames(data, original_base, save_root):
    for frame_paths, _ in data:
        for frame_path in frame_paths:
            rel_path = os.path.relpath(frame_path, original_base)
            save_path = os.path.join(save_root, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not os.path.exists(save_path):
                crop_upper_body(frame_path, save_path)

# ==== EXTRACT CLIP FEATURES ====
@torch.no_grad()
def extract_clip_features_multiframe(data, original_base, cropped_base):
    features, labels = [], []
    for frame_paths, label in data:
        img_tensors = []
        for frame_path in frame_paths[:MAX_FRAMES]:
            rel_path = os.path.relpath(frame_path, original_base)
            cropped_path = os.path.join(cropped_base, rel_path)
            if not os.path.exists(cropped_path): continue
            image = Image.open(cropped_path).convert("RGB")
            img = preprocess(image).unsqueeze(0).to(device)
            img_tensors.append(img)
        if not img_tensors:
            continue
        imgs = torch.cat(img_tensors)
        img_feats = clip_model.encode_image(imgs)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        video_feat = img_feats.mean(dim=0, keepdim=True)
        features.append(video_feat.cpu())
        labels.append(label)
    return torch.cat(features), torch.tensor(labels)

# ==== TRAIN + EVALUATE ====
def train_and_evaluate(train_features, train_labels, val_features, val_labels, test_features, test_labels, label_map):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train_features.numpy(), train_labels.numpy())

    val_preds = clf.predict(val_features.numpy())
    test_preds = clf.predict(test_features.numpy())

    print("Validation Accuracy:", accuracy_score(val_labels.numpy(), val_preds))
    print("Test Accuracy:", accuracy_score(test_labels.numpy(), test_preds))
    print("\nClassification Report (Test):")
    print(classification_report(test_labels.numpy(), test_preds, target_names=label_map.keys()))

    conf = confusion_matrix(test_labels.numpy(), test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title("Confusion Matrix")
    plt.show()

# ==== MAIN PIPELINE ====
if __name__ == "__main__":
    print("Loading datasets...")
    train_data, label_map = load_samples_flat(DATASET_PATHS["train"], TARGET_CLASSES)
    val_data, _ = load_samples_flat(DATASET_PATHS["val"], TARGET_CLASSES)
    test_data, _ = load_samples_flat(DATASET_PATHS["test"], TARGET_CLASSES)

    print("Cropping frames with YOLOv8...")
    crop_all_frames(train_data, DATASET_PATHS["train"], CROPPED_DIR)
    crop_all_frames(val_data, DATASET_PATHS["val"], CROPPED_DIR)
    crop_all_frames(test_data, DATASET_PATHS["test"], CROPPED_DIR)

    print("Extracting CLIP features...")
    train_features, train_labels = extract_clip_features_multiframe(train_data, DATASET_PATHS["train"], CROPPED_DIR)
    val_features, val_labels = extract_clip_features_multiframe(val_data, DATASET_PATHS["val"], CROPPED_DIR)
    test_features, test_labels = extract_clip_features_multiframe(test_data, DATASET_PATHS["test"], CROPPED_DIR)

    print("Training classifier and evaluating...")
    train_and_evaluate(train_features, train_labels, val_features, val_labels, test_features, test_labels, label_map)
