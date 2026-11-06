import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="combine Metacarpal and Phalanx mask to make Ray mask")
    parser.add_argument('--root', type=str, required=True, help="result folder")
    parser.add_argument('--meta_dir', type=str, required=True, help="Metacarpal mask folder")
    parser.add_argument('--phal_dir', type=str, required=True, help="Phalanx mask folder")
    parser.add_argument('--save_name', type=str, required=True, help="folder and CSV name to save")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    save_path = os.path.join(args.root, args.save_name)
    csv_dir = os.path.join(args.root, "list")
    save_csv_path = os.path.join(csv_dir, f"{args.save_name}.csv")
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    meta_dict = {}  # { 'image ID' : 'metacarpal mask path' }
    phal_dict = {}  # { 'image ID' : 'phalanx mask path' }
    group_dict = {} # { 'image ID' : 'group name (F_20_45 etc.)' }

    print("Scanning folders to map file lists and groups...")

    for root_dir, _, files in os.walk(args.meta_dir):
        group_name = os.path.basename(root_dir) 
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                img_id = os.path.splitext(file)[0]
                meta_dict[img_id] = os.path.join(root_dir, file)
                group_dict[img_id] = group_name

    for root_dir, _, files in os.walk(args.phal_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                img_id = os.path.splitext(file)[0]
                phal_dict[img_id] = os.path.join(root_dir, file)

    common_ids = set(meta_dict.keys()) & set(phal_dict.keys())
    print(f"[*] Meta masks: {len(meta_dict)} | Phal masks: {len(phal_dict)}")
    print(f"[*] Common masks (Ray synthesis targets): {len(common_ids)}\n")

    successful_records = []

    for img_id in tqdm(common_ids, desc="Merging Ray Masks"):
        try:
            m_path = meta_dict[img_id]
            p_path = phal_dict[img_id]

            m_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            p_mask = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)

            if m_mask is None or p_mask is None:
                raise ValueError("cannot read mask images")
            if m_mask.shape != p_mask.shape:
                p_mask = cv2.resize(p_mask, (m_mask.shape[1], m_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            ray_mask = cv2.bitwise_or(m_mask, p_mask)

            group = group_dict.get(img_id, "Unknown")
            out_dir = os.path.join(save_path, group)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"{img_id}.png")
            cv2.imwrite(out_file, ray_mask)

            successful_records.append({'id': img_id, 'group': group})

        except Exception as e:
            print(f"\n[!] {img_id} skipped - reason: {e}")

    df = pd.DataFrame(successful_records, columns=['id', 'group'])
    df.to_csv(save_csv_path, index=False)