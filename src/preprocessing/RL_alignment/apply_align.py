import os
import csv
import cv2
from tqdm import tqdm

INPUT_ROOT = "./data/anon_data_flatten_cropped"
OUTPUT_ROOT = "./data/anon_data_flatten_cropped_aligned"
CSV_PATH = "./data/predictions_add.csv"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

saved = 0
skipped = 0

subdirs = [
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d))
]

multi_age_structure = len(subdirs) > 0

for row in tqdm(rows, desc="Aligning images"):

    final = row["final"].strip()
    if final not in ["left", "right"]:
        skipped += 1
        continue

    age = row["age_group"]
    fname = row["image_id"]

    if multi_age_structure:
        in_path = os.path.join(INPUT_ROOT, age, fname)
        out_dir = os.path.join(OUTPUT_ROOT, age)
    else:
        in_path = os.path.join(INPUT_ROOT, fname)
        out_dir = OUTPUT_ROOT
        age = None

    if multi_age_structure:
        os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        skipped += 1
        continue

    if final == "left":
        img = cv2.flip(img, 1)

    cv2.imwrite(os.path.join(out_dir, fname), img)
    saved += 1

print("\n========== SUMMARY ==========")
print(f"Saved (aligned): {saved}")
print(f"Skipped (no final label / read fail): {skipped}")
