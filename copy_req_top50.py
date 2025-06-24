import os
import shutil

SOURCE_DIR = "PHOENIX-2014-T/features/fullFrame-210x260px/train"
DEST_DIR = "PHOENIX-2014-T/features/fullFrame-210x260px/train_top50"
NAMES_FILE = "subset_video_names.txt"

# Create destination if not exist
os.makedirs(DEST_DIR, exist_ok=True)

# Read video folder names
with open(NAMES_FILE, "r") as f:
    names = [line.strip() for line in f.readlines()]

# Copy each folder
missing = []
copied = 0
for name in names:
    src = os.path.join(SOURCE_DIR, name)
    dst = os.path.join(DEST_DIR, name)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied += 1
    else:
        missing.append(name)

print(f"✅ Copied {copied} frame folders to: {DEST_DIR}")
if missing:
    print(f"⚠️ Missing {len(missing)} folders (not found): {missing[:5]}...")
