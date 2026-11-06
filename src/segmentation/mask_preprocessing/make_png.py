import json
import cv2
import numpy as np
import os
import urllib.parse
import argparse
import pandas as pd
from tqdm import tqdm
"""
 make hand craft segmentation mask images from json files
"""
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, nargs='+', default=None)
parser.add_argument('--ROOT', type=str, default="./data/seg")
parser.add_argument('--save_png_path', type=str, default="masks", help="mask save folder name")
parser.add_argument('--target_labels', type=str, nargs='+', default=[
    ], help="Enter labels to be converted to masks separated by spaces")
args = parser.parse_args()

JSON_FILES = args.json_path
ROOT = args.ROOT
OUTPUT_DIR = os.path.join(ROOT, args.save_png_path)
OUTPUT_CSV = os.path.join(ROOT, "list/masks.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

TARGET_LABELS = args.target_labels
all_processed_names = []
total_success_count = 0
total_duplicate_count = 0
seen_image_ids = set()

for json_file in JSON_FILES:
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} was not found; skipping it.")
        continue

    print(f"\nProcessing JSON: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    success_count = 0
    duplicate_count = 0
    for task in tqdm(data, desc="Generating Masks"):
        if not task.get('annotations') or not task.get('annotations')[0].get('result'):
            continue
        
        raw_name = task.get('file_upload')
        if not raw_name: continue
        
        basename = os.path.basename(urllib.parse.unquote(raw_name))
        origin_img_name = basename.split('-')[-1]

        if origin_img_name in seen_image_ids:
            duplicate_count += 1
            continue
        seen_image_ids.add(origin_img_name)
        mask_name = os.path.splitext(origin_img_name)[0] + '.png'
        mask_path = os.path.join(OUTPUT_DIR, mask_name)

        results = task['annotations'][0]['result']
        if not results: continue

        w = results[0]['original_width']
        h = results[0]['original_height']
        mask = np.zeros((h, w), dtype=np.uint8)
        draw_flag = False

        for result in results:
            if result['type'] == 'polygonlabels':
                current_labels = result['value'].get('polygonlabels', [])
                if any(label in TARGET_LABELS for label in current_labels):
                    draw_flag = True
                    points = result['value']['points']
                    abs_points = [[int(p[0]*w/100.0), int(p[1]*h/100.0)] for p in points]
                    abs_points = np.array(abs_points, dtype=np.int32)
                    cv2.fillPoly(mask, [abs_points], 255)

        if draw_flag:
            cv2.imwrite(mask_path, mask)
            all_processed_names.append(origin_img_name)
            success_count += 1
    total_duplicate_count += duplicate_count    
    total_success_count += success_count
    print(f"Done: {success_count} masks created from this JSON (duplicates skipped: {duplicate_count}).")

new_df = pd.DataFrame(all_processed_names, columns=['image_id'])
if os.path.isfile(OUTPUT_CSV):
    old_df = pd.read_csv(OUTPUT_CSV)
    combined_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['image_id'], keep='last')
    combined_df.to_csv(OUTPUT_CSV, index=False)
else:
    new_df.drop_duplicates(subset=['image_id']).to_csv(OUTPUT_CSV, index=False)

print(f"\nAll process complete!")
print(f"Total successfully processed: {total_success_count}")
print(f"Total duplicates skipped: {total_duplicate_count}")
print(f"Final CSV saved to {OUTPUT_CSV}")
