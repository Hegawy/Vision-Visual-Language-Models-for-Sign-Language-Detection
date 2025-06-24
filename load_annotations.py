import pandas as pd
from collections import Counter
import os

# --- Parameters ---
TOP_N = 50
base_dir = "PHOENIX-2014-T"
annotation_file = os.path.join(base_dir, "annotations/manual/PHOENIX-2014-T.train.corpus.csv")
output_dir = os.path.join(base_dir, "features_clip_sla_top50")
os.makedirs(output_dir, exist_ok=True)

# --- Load the CSV ---
df = pd.read_csv(annotation_file, sep='|', header=None)
df.columns = ['name', 'video', 'start', 'end', 'speaker', 'orth', 'translation']

# --- Get top-N frequent sentences ---
sentence_counts = Counter(df['orth'])
top_sentences = [sent for sent, _ in sentence_counts.most_common(TOP_N)]

# --- Filter dataset to top-N sentences ---
df_filtered = df[df['orth'].isin(top_sentences)].reset_index(drop=True)

# --- Save filtered annotations ---
subset_csv = os.path.join(output_dir, "subset_annotations_top50.csv")
df_filtered.to_csv(subset_csv, index=False)

# --- Save video folder names for copying frames ---
subset_names_file = os.path.join(output_dir, "subset_video_names.txt")
with open(subset_names_file, "w") as f:
    for name in df_filtered["name"].unique():
        f.write(name + "\n")

# --- Save label map (sentence → class index) ---
label_map = {sent: idx for idx, sent in enumerate(sorted(df_filtered["orth"].unique()))}
label_map_path = os.path.join(output_dir, "label_map.txt")
with open(label_map_path, "w") as f:
    for sent, idx in label_map.items():
        f.write(f"{sent}\t{idx}\n")

# --- Summary ---
print(f"✅ Selected Top-{TOP_N} Sentences.")
print(f"📊 Total samples: {len(df_filtered)} from {len(df_filtered['orth'].unique())} unique sentences.")
print(f"✅ Saved filtered annotation to: {subset_csv}")
print(f"✅ Saved video names to: {subset_names_file}")
print(f"✅ Saved label map to: {label_map_path}")
