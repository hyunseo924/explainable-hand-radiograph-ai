import os
import argparse
import numpy as np
import pandas as pd
import shutil
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLOv8 Bounding Box Detection Training & Inference")
parser.add_argument('--root',type=str,default="./yolo",help="results root(yolo_dataset,checkpoints,inference results)")
parser.add_argument('--img_dir', type=str, default="./data/anon_data_flatten_cropped_aligned_masking", help="origin X-ray image directory")
parser.add_argument('--mask_dir', type=str, default="./data/seg/masks", help="PNG mask directory")
parser.add_argument('--mask_csv', type=str, default="./data/list/masks.csv", help="mask list CSV file")
parser.add_argument('--removal_csv', type=str, default="./data/list/Removal.csv", help="image list to exclude during inference")
parser.add_argument('--epochs', type=int, default=50, help="YOLO training epochs")
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for YOLO training")
parser.add_argument('--class_name', type=str, default="target", help="single YOLO class name written to data.yaml")
args = parser.parse_args()
root = args.root
yolo_data_dir = os.path.join(root,"yolo_dataset")
save_dir = os.path.join(root,"checkpoints")
infer_save_dir = os.path.join(root,"yolo_inference_results")

dirs = {
    "train_img": os.path.join(yolo_data_dir, "images", "train"),
    "val_img": os.path.join(yolo_data_dir, "images", "val"),
    "train_lbl": os.path.join(yolo_data_dir, "labels", "train"),
    "val_lbl": os.path.join(yolo_data_dir, "labels", "val")
}
for path in dirs.values():
    os.makedirs(path, exist_ok=True)

df = pd.read_csv(args.mask_csv)
image_ids = df['image_id'].tolist()
train_imgs, val_imgs = train_test_split(image_ids, test_size=0.1, random_state=42)

print(f"Split {len(image_ids)} images into Train: {len(train_imgs)} and Val: {len(val_imgs)}")

group_dirs = ["F_20_45", "F_55_80"]

def get_yolo_bbox(mask_np, margin=15):
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0:
        return None
    
    H, W = mask_np.shape
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    x_min = max(0, x_min - margin)
    x_max = min(W, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(H, y_max + margin)
    
    x_center = ((x_min + x_max) / 2.0) / W
    y_center = ((y_min + y_max) / 2.0) / H
    box_width = (x_max - x_min) / W
    box_height = (y_max - y_min) / H
    
    return [x_center, y_center, box_width, box_height]

def prepare_yolo_data(img_list, split_type):
    print(f"Converting {split_type.upper()} data...")
    img_save_dir = dirs[f"{split_type}_img"]
    lbl_save_dir = dirs[f"{split_type}_lbl"]
    
    for img_name in tqdm(img_list, leave=False):
        img_name = str(img_name)
        img_name = os.path.splitext(img_name)[0]
        img_path = None
        for g in group_dirs:
            temp_path = os.path.join(args.img_dir, g, f"{img_name}.jpg")
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if img_path is None: continue
            
        mask_path = os.path.join(args.mask_dir, img_name + '.png')
        if not os.path.exists(mask_path): continue
            
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        
        bbox = get_yolo_bbox(mask)
        if bbox is None: continue
            
        dest_img_path = os.path.join(img_save_dir, f"{img_name}.jpg")
        if os.path.exists(dest_img_path) or os.path.islink(dest_img_path):
            os.remove(dest_img_path)
            
        os.symlink(img_path, dest_img_path)

        
        dest_lbl_path = os.path.join(lbl_save_dir, f"{img_name}.txt")
        with open(dest_lbl_path, 'w') as f:
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

prepare_yolo_data(train_imgs, "train")
prepare_yolo_data(val_imgs, "val")

yaml_path = os.path.join(yolo_data_dir, "data.yaml")
yaml_data = {
    'path': yolo_data_dir,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': [args.class_name]
}
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False)
    
print(f" YOLO dataset preparation complete (Location: {yolo_data_dir})")

print("\nYOLOv8 Nano model training started")
model = YOLO('yolov8n.pt') 

results = model.train(
    data=yaml_path,
    epochs=args.epochs,
    imgsz=1024,
    lr0=args.lr,
    batch=args.batch,
    project=save_dir,
    name="yolov8n_bone_det",
    exist_ok=True,
    patience=20,
    mosaic=0.0,
    close_mosaic=0,
)

best_weight_path = os.path.join(save_dir, 'yolov8n_bone_det', 'weights', 'best.pt')
print(f"\nYOLO training complete. Best weights: {best_weight_path}")

print("\nTraining completed model for full X-ray inference and bounding box visualization...")

best_model = YOLO(best_weight_path)
removal_df = pd.read_csv(args.removal_csv)
removal_set = set(removal_df['id'].values)
print(f"Excluding {len(removal_set)} images listed in Removal.csv from inference.")
os.makedirs(infer_save_dir, exist_ok=True)

infer_count = 0
for group in group_dirs:
    group_path = os.path.join(args.img_dir, group)
    if not os.path.exists(group_path):
        continue
        
    group_save_dir = os.path.join(infer_save_dir, group)
    os.makedirs(group_save_dir, exist_ok=True)
    
    image_files = sorted(f for f in os.listdir(group_path) if f.lower().endswith(('.jpg', '.png')))
    
    for img_file in tqdm(image_files, desc=f"{group} Inference"):
        if img_file in removal_set:
            continue
            
        img_path = os.path.join(group_path, img_file)
        infer_results = best_model.predict(img_path, verbose=False)        
        annotated_img = infer_results[0].plot() 
        save_path = os.path.join(group_save_dir, img_file)
        cv2.imwrite(save_path, annotated_img)
        infer_count += 1

print(f"\nInference complete. A total of {infer_count} images have bounding boxes drawn and saved to the following location:")
print(f"{infer_save_dir}")
