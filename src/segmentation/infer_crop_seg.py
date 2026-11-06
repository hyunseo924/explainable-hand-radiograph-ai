import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from transformers import SamModel, SamProcessor, SamConfig
from ultralytics import YOLO
parser = argparse.ArgumentParser(description="Two-Stage (YOLO Crop + SAM/U-Net) Segmentation Inference Pipeline")
parser.add_argument('--img_dir', type=str, default="./data/anon_data_flatten_cropped_aligned_masking", help="original X-ray directory")
parser.add_argument('--root', type=str, required=True, help="results save root")
parser.add_argument('--model_type', type=str, required=True, choices=['unet', 'sam'], help="model architecture for segmentation stage (unet or sam)")
parser.add_argument('--sam_path', type=str, required=True, help="Segmentation model weight path(.pt)")
parser.add_argument('--yolo_path', type=str, default = "./yolo/checkpoints/yolov8n_bone_det/weights/best.pt",help="YOLOv8 detection model weight path(.pt)")
parser.add_argument('--sam_id', type=str, default='facebook/sam-vit-base', choices=["facebook/sam-vit-base","wanglab/medsam-vit-base"])
parser.add_argument('--save_name', required=True, type=str, help="results save folder name (under root)")
parser.add_argument('--removal_csv', type=str, default="./data/list/Removal.csv")
parser.add_argument('--fail_removal',type=str,default=None,help='save path for failure case (CSV)')
parser.add_argument('--invert',action='store_true',help='invert mask application logic')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = os.path.join(args.root, args.save_name, f"masked_xrays_{args.save_name}")
mask_dir = os.path.join(args.root, args.save_name, f"pseduo_masks_{args.save_name}")
os.makedirs(output_dir, exist_ok=True)

print("[1/2] loading YOLOv8...")
yolo_model = YOLO(args.yolo_path)

print(f"[2/2] loading {args.model_type.upper()} ... from ({args.sam_path})")
processor = None

if args.model_type == 'unet':
    INFERENCE_SIZE = 512
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(args.sam_path, map_location=DEVICE))
    model.eval()
    
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=INFERENCE_SIZE),
        A.PadIfNeeded(min_height=INFERENCE_SIZE, min_width=INFERENCE_SIZE, border_mode=0, value=0),
        A.CLAHE(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
elif args.model_type == 'sam':
    INFERENCE_SIZE = 1024
    processor = SamProcessor.from_pretrained(args.sam_id)
    config = SamConfig.from_pretrained(args.sam_id)
    model = SamModel(config).to(DEVICE)
    model.load_state_dict(torch.load(args.sam_path, map_location=DEVICE), strict=False)
    model.eval()
    
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=INFERENCE_SIZE),
        A.PadIfNeeded(min_height=INFERENCE_SIZE, min_width=INFERENCE_SIZE, border_mode=0, value=0),
        A.CLAHE(p=1.0)
    ])

def undo_pad_and_resize(padded_mask_np, orig_h, orig_w, max_size):
    scale = max_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    pad_y = (max_size - new_h) // 2
    pad_x = (max_size - new_w) // 2
    
    unpadded_mask = padded_mask_np[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    
    restored_mask = cv2.resize(unpadded_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (restored_mask > 0.5).astype(np.uint8) * 255

removal_df = pd.read_csv(args.removal_csv)
removal_set = set(removal_df['id'].values)

group_dirs = ["F_20_45", "F_55_80"]
infer_count = 0
no_box_cnt = 0
small_box_cnt = 0
fail_record = []
print("\nStart two-stage inference...")
with torch.no_grad():
    for group in group_dirs:
        group_path = os.path.join(args.img_dir, group)
        if not os.path.exists(group_path): continue
        
        save_mask_dir = os.path.join(mask_dir, group)
        save_final_dir = os.path.join(output_dir, group)
        os.makedirs(save_mask_dir, exist_ok=True)
        os.makedirs(save_final_dir, exist_ok=True)
        
        image_files = sorted(f for f in os.listdir(group_path) if f.lower().endswith(('.jpg', '.png')))
        
        for img_file in tqdm(image_files, desc=f"{group} Inference"):
            if img_file in removal_set: continue
                
            img_path = os.path.join(group_path, img_file)
            img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            original_h, original_w = img_np.shape[:2]
            
            yolo_results = yolo_model.predict(img_np, verbose=False)
            boxes = yolo_results[0].boxes
            
            if len(boxes) == 0:
                blank_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                cv2.imwrite(os.path.join(save_mask_dir, os.path.splitext(img_file)[0] + ".png"), blank_mask)
                cv2.imwrite(os.path.join(save_final_dir, img_file), np.zeros_like(img_np))
                no_box_cnt += 1
                if args.fail_removal is not None:
                    fail_record.append({'id': img_file, 'group': group,'why':'no_box'})
                continue
                
            best_box = boxes[np.argmax(boxes.conf.cpu().numpy())]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)
            
            cropped_img = img_np[y1:y2, x1:x2]
            crop_h, crop_w = cropped_img.shape[:2]
            
            if crop_h < 10 or crop_w < 10:
                small_box_cnt += 1
                if args.fail_removal is not None:
                    fail_record.append({'id': img_file, 'group': group,'why':'small_box'})
                continue
            
            augmented = val_transform(image=cropped_img)
            transformed_img = augmented["image"]
            
            if args.model_type == 'unet':
                input_tensor = transformed_img.unsqueeze(0).to(DEVICE)
                logits = model(input_tensor)
                prob_mask = torch.sigmoid(logits).squeeze().cpu().numpy()
                
            elif args.model_type == 'sam':
                h, w = transformed_img.shape[:2]
                prompt_box = [0, 0, w, h]
                
                inputs = processor(transformed_img, input_boxes=[[prompt_box]], return_tensors="pt").to(DEVICE)
                outputs = model(**inputs, multimask_output=False)
                
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
                )
                prob_mask = masks[0][0][0].numpy().astype(np.float32)

            restored_crop_mask = undo_pad_and_resize(prob_mask, crop_h, crop_w, INFERENCE_SIZE)
            
            final_binary_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            final_binary_mask[y1:y2, x1:x2] = restored_crop_mask
 
            final_binary_mask =cv2.morphologyEx(final_binary_mask, cv2.MORPH_CLOSE, (9, 9))
            final_binary_mask = cv2.morphologyEx(final_binary_mask, cv2.MORPH_OPEN, (9, 9))
            #contours, _ = cv2.findContours(final_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #contour_mask = np.zeros_like(final_binary_mask)
            #cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
            #final_binary_mask = contour_mask

            if args.invert:
                applied_mask = final_binary_mask
            else: 
                applied_mask = cv2.bitwise_not(final_binary_mask)
            mask_3ch = cv2.cvtColor(applied_mask, cv2.COLOR_GRAY2RGB)
            final_masked_img = cv2.bitwise_and(img_np, mask_3ch)
            
            basename_no_ext = os.path.splitext(img_file)[0]
            cv2.imwrite(os.path.join(save_mask_dir, basename_no_ext + ".png"), final_binary_mask)
            cv2.imwrite(os.path.join(save_final_dir, img_file), cv2.cvtColor(final_masked_img, cv2.COLOR_RGB2BGR))
            
            infer_count += 1

if args.fail_removal is not None and (small_box_cnt+no_box_cnt) > 0:
    fail_df = pd.DataFrame(fail_record)
    
    fail_df = fail_df.drop_duplicates()
    
    os.makedirs(os.path.dirname(args.fail_removal), exist_ok=True)
    
    if os.path.exists(args.fail_removal):
        fail_df.to_csv(args.fail_removal, mode='a', header=False, index=False)
    else:
        fail_df.to_csv(args.fail_removal, mode='w', header=True, index=False)
elif (small_box_cnt+no_box_cnt) == 0:
    print("\n all inference successful. No failure case to record.")


print(f"\nfinish! success case: {infer_count} | small box : {small_box_cnt} | no box : {no_box_cnt} / {small_box_cnt+no_box_cnt+infer_count}")
print(f"inference results: {output_dir}")
