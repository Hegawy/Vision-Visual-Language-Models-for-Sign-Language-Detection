import os
import shutil
from tqdm import tqdm

# --- Paths ---
asl_train_dir = "/Users/hegawy/Desktop/Final Project Bachelor/ASL alphabet Dataset/asl_alphabet_train/asl_alphabet_train"
lexset_train_dir = "/Users/hegawy/Desktop/Final Project Bachelor/Synthetic ASL/Train_Alphabet"
lexset_test_dir = "/Users/hegawy/Desktop/Final Project Bachelor/Synthetic ASL/Test_Alphabet"  # optional
combined_dir = "/Users/hegawy/Desktop/Final Project Bachelor/combined_dataset"

# --- Create target structure ---
all_classes = set(os.listdir(asl_train_dir)) | set(os.listdir(lexset_train_dir)) | set(os.listdir(lexset_test_dir))

for class_name in all_classes:
    os.makedirs(os.path.join(combined_dir, class_name), exist_ok=True)

def copy_images(src_dir, prefix):
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        dst_class_dir = os.path.join(combined_dir, class_name)
        for fname in tqdm(os.listdir(class_path), desc=f"Copying {prefix}_{class_name}"):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(class_path, fname)
                new_name = f"{prefix}_{fname}"
                dst_path = os.path.join(dst_class_dir, new_name)
                shutil.copy2(src_path, dst_path)

# --- Copy from ASL train ---
copy_images(asl_train_dir, prefix="asl")

# --- Copy from Lexset train ---
copy_images(lexset_train_dir, prefix="lexset")

# --- Optionally include Lexset test images ---
copy_images(lexset_test_dir, prefix="lexsetTest")

print("✅ Merged dataset created at:", combined_dir)
