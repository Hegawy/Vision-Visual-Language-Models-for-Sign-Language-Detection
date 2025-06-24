# copy_frame_folders.py

import os
import shutil
import pandas as pd

# Load the subset file created earlier
subset_csv = 'PHOENIX-2014-T/annotations/manual/subsets/train_sentence_subset.csv'
df = pd.read_csv(subset_csv, delimiter='|', header=None)
df.columns = ['name', 'video', 'start', 'end', 'speaker', 'orth', 'translation']

# Define original and target video frame directories
source_dir = 'PHOENIX-2014-T/features/fullFrame-210x260px/train/'
target_dir = 'PHOENIX-2014-T/features/fullFrame-210x260px/train_subset/'

# Create target directory
os.makedirs(target_dir, exist_ok=True)

# Copy only necessary folders
video_names = df['name'].unique()
copied = 0

for video_name in video_names:
    src = os.path.join(source_dir, video_name)
    dst = os.path.join(target_dir, video_name)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied += 1
    else:
        print(f"⚠️ Missing: {src}")

print(f"✅ Copied {copied} video frame folders to: {target_dir}")
