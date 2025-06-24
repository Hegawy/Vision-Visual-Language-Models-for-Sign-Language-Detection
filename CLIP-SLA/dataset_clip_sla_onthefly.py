import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# CLIP ViT-B/32 preprocessing
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]),
])

class CLIPSLAOnTheFlyDataset(Dataset):
    def __init__(self, root_dir, label_map, max_len=64):
        self.samples = []
        self.label_map = label_map
        self.max_len = max_len

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path): continue

            for sample in os.listdir(label_path):
                sample_path = os.path.join(label_path, sample)
                if os.path.isdir(sample_path):
                    self.samples.append((sample_path, label_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label_name = self.samples[idx]
        label_id = self.label_map[label_name]

        frames = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".jpg") or f.endswith(".png")
        ])[:self.max_len]

        images = []
        for frame in frames:
            image = Image.open(os.path.join(folder, frame)).convert("RGB")
            tensor = clip_preprocess(image)
            images.append(tensor)

        # Padding
        while len(images) < self.max_len:
            images.append(torch.zeros(3, 224, 224))

        video_tensor = torch.stack(images)  # Shape: [T, 3, 224, 224]
        return video_tensor, label_id
