import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from transformers import SamModel, SamProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Bone Segmentation Fine-Tuning")
parser.add_argument('--img_dir', type=str, default="./data/anon_data_flatten_cropped_aligned_masking", help="origin X-ray directory")
parser.add_argument('--mask_dir', type=str, default="./data/seg/masks", help="PNG mask directory")
parser.add_argument('--mask_csv', type=str, default="./data/list/masks.csv", help="mask list CSV file")
parser.add_argument('--model_type', type=str, default='sam', choices=['unet', 'sam'], help="model type to fine-tune (unet or sam)")
parser.add_argument('--sam_id', type=str, default='facebook/sam-vit-base', help='"facebook/sam-vit-base" or "wanglab/medsam-vit-base"')
parser.add_argument('--save_dir', type=str, default="./data/models/5_masking_ablation/checkpoints", help="model save directory")
parser.add_argument('--save_name',type=str, required = True)
parser.add_argument('--epoch',type=int,default=25)
parser.add_argument('--lr',type=float,default=None)
args = parser.parse_args()
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{DEVICE}] : fine-tuning {args.model_type.upper()}")

os.makedirs(args.save_dir, exist_ok=True)
df = pd.read_csv(args.mask_csv)

image_ids = df['image_id'].tolist() 
train_imgs, val_imgs = train_test_split(image_ids, test_size=0.1, random_state=42)

print(f"Split {len(image_ids)} images into Train: {len(train_imgs)} and Val: {len(val_imgs)}")

def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0:
        return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    margin_x = np.random.randint(10,30)
    margin_y = np.random.randint(10,30)

    H, W = ground_truth_map.shape
    x_min = max(0, x_min - margin_x)
    x_max = min(W, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(H, y_max + margin_y)
    
    return [x_min, y_min, x_max, y_max]

class BoneSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, model_type, transform=None, processor=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.model_type = model_type
        self.transform = transform
        self.processor = processor
        self.group = ["F_20_45","F_55_80"]
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = str(self.image_list[idx])
        for g in self.group:
            img_path = os.path.join(self.image_dir,g ,img_name)
            if os.path.exists(img_path):
                break
        
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)

        # crop!! 
        x1, y1, x2, y2 = get_bounding_box(mask)
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        if self.model_type == 'unet':
            if self.transform:
                augmented = self.transform(image=cropped_image, mask=cropped_mask) 
                cropped_image = augmented["image"]
                cropped_mask = augmented["mask"]
            cropped_mask =cropped_mask.unsqueeze(0)
            return cropped_image, cropped_mask

        elif self.model_type == 'sam':

            if self.transform:
                augmented = self.transform(image=cropped_image, mask=cropped_mask)
                cropped_image = augmented["image"]
                cropped_mask = augmented["mask"]
            
            h, w = cropped_mask.shape
            prompt_box = [0, 0, w, h]
            
            inputs = self.processor(cropped_image, input_boxes=[[prompt_box]], return_tensors="pt")
                    
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = torch.tensor(cropped_mask, dtype=torch.float32)
            
            return inputs

if args.model_type == 'unet':
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8), 
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0),
        A.CLAHE(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
else:
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=1024),
        A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=1.0)
    ])
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=1024),
        A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0),
        A.CLAHE(p=1.0)
    ])
processor = None
if args.model_type == 'sam':
    processor = SamProcessor.from_pretrained(args.sam_id)
    model = SamModel.from_pretrained(args.sam_id).to(DEVICE)
    
    for name, param in model.vision_encoder.named_parameters():
        param.requires_grad = False
    for name, param in model.prompt_encoder.named_parameters():
        param.requires_grad = False
    print("SAM vision and prompt encoders frozen; training the mask decoder only.")
else:
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1).to(DEVICE)

train_loader = DataLoader(BoneSegDataset(args.img_dir, args.mask_dir, train_imgs, args.model_type, train_transform, processor), batch_size=2, shuffle=True)
val_loader = DataLoader(BoneSegDataset(args.img_dir, args.mask_dir, val_imgs, args.model_type, val_transform, processor), batch_size=2, shuffle=False)

criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
lr = 1e-4 if args.model_type == 'sam' else 3e-4
if args.lr is not None:
    lr=args.lr
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

EPOCHS = args.epoch
best_val_loss = float('inf')

print("\n > . < training start")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader,desc=f"Epoch {epoch}/{EPOCHS} - Training", leave=False):
        optimizer.zero_grad()
        
        if args.model_type == 'unet':
            images, masks = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
        else: # SAM
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)
            ground_truth_masks = batch["ground_truth_mask"].to(DEVICE).unsqueeze(1) # [B, 1, H, W]
            
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_masks = outputs.pred_masks.squeeze(1) 
            pred_masks = torch.nn.functional.interpolate(pred_masks, size=ground_truth_masks.shape[-2:], mode="bilinear", align_corners=False)
            
            loss = criterion(pred_masks, ground_truth_masks)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if args.model_type == 'unet':
                images, masks = batch[0].to(DEVICE), batch[1].to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
            else:
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_boxes = batch["input_boxes"].to(DEVICE)
                ground_truth_masks = batch["ground_truth_mask"].to(DEVICE).unsqueeze(1)
                
                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                pred_masks = torch.nn.functional.interpolate(outputs.pred_masks.squeeze(1), size=ground_truth_masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(pred_masks, ground_truth_masks)
                
            val_loss += loss.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    save_name=args.save_name

    print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
    
    save_path = os.path.join(args.save_dir, f"{save_name}.pth")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"  <- Best Model Saved ({save_name})")
    else:
        print()
rename_path = os.path.join(args.save_dir,f"{save_name}_{best_val_loss:.4f}.pth")
print("\n training complete. Best model saved as:", rename_path)
