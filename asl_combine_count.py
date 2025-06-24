import os
from collections import defaultdict

# --- Path to your combined dataset ---
data_dir = "/Users/hegawy/Desktop/Final Project Bachelor/combined_dataset"  # 👈 update this

# --- Count images per class ---
class_counts = defaultdict(int)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        class_counts[class_name] = count

# --- Sort and print counts ---
sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n📊 Class Distribution in: {data_dir}\n")
for cls, count in sorted_counts:
    print(f"{cls:<12}: {count} images")

# --- Optional: print imbalance warning ---
avg = sum(class_counts.values()) / len(class_counts)
print(f"\n📉 Average per class: {avg:.2f}")
print("⚠️ Classes with fewer than 50% of avg:")

for cls, count in sorted_counts:
    if count < 0.5 * avg:
        print(f" - {cls} ({count})")
