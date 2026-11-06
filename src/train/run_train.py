import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import argparse
import cv2
import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(model_name, num_classes=2, pretrained=True):
    model = None
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name =="densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)    
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name.startswith("vit"):
        if model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None 
            model = models.vit_b_16(weights=weights)
        elif model_name == "vit_b_32":
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vit_b_32(weights=weights)
        
        if model is not None:
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    if model is None:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model
 
def prepare_dataframe(root_dir,split_seed=42,removal_files=None):
    data = []
    groups = {"F_20_45": 0, "F_55_80": 1}

    removal_ids = set()
    if removal_files is not None:
        if isinstance(removal_files,str):
            removal_files = [removal_files]
        for r_file in removal_files:
            if os.path.exists(r_file):
                removal_ids.update(pd.read_csv(r_file)["id"].values)
            else:
                print(f"warning: File not found: {r_file}")
    
    for folder_name, label in groups.items():
        folder_path = os.path.join(root_dir, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            if removal_files is not None:
                fname = os.path.basename(f)
                if fname in removal_ids:
                    continue
            data.append({"image_path": f, "label": label})
            
    df = pd.DataFrame(data)
    print(f"Total samples after removal: {len(df)}")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=split_seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=split_seed)

    return train_df, val_df, test_df

class HandXrayDataset(Dataset):
    def __init__(self, dataframe, img_size, resize_mode, train=True):
        self.df = dataframe.reset_index(drop=True)
        self.train = train
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if self.train:
            aug_list = [
                A.Rotate(limit=7), 
                A.ColorJitter(brightness=0.15, contrast=0.15, p=0.5),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
            ]
        else:
            aug_list = [] 

        if resize_mode == "random" and self.train:
            base_preprocess = [A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0))]
        elif resize_mode == "padded":
            base_preprocess = [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REPLICATE)
            ]
        elif resize_mode == "zeropad":
            base_preprocess = [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0))
            ]
        elif resize_mode =="reflect":
            base_preprocess = [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101) 
            ]
        else:
            base_preprocess = [A.Resize(img_size, img_size)]

        self.transform = A.Compose([
            *aug_list,           
            *base_preprocess,    
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img_np = np.array(img)
            
            augmented = self.transform(image=img_np)
            img = augmented['image']
            
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return img, label
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
            return self.__getitem__(0)

def run_training(args,split_seed=42):

    seed_everything(args.seed)
    os.makedirs(args.save_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_model_name = args.save_name if args.save_name else f"{args.model_name}_seed{args.seed}"
    SAVE_MODEL_PATH = os.path.join(args.save_root, f"best_{save_model_name}.pth")
    LOG_CSV_PATH = os.path.join(args.save_root,'log', f"training_{save_model_name}_log.csv")
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)

    train_df, val_df, test_df = prepare_dataframe(args.data_root, split_seed, removal_files=args.removal)
    train_loader = DataLoader(HandXrayDataset(train_df, args.img_size, args.resize_mode, train=True), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(HandXrayDataset(val_df, args.img_size, args.resize_mode, train=False), batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = get_model(args.model_name).to(device)
    if args.use_focal_loss: criterion = FocalLoss(gamma=args.focal_gamma)
    else: criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    def evaluate(loader):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(loader.dataset)
        all_probs_np, all_labels_np = np.array(all_probs), np.array(all_labels)
        
        best_acc, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.1, 0.91, 0.01):
            preds = (all_probs_np >= thresh).astype(int)
            acc = np.mean(preds == all_labels_np)
            if acc > best_acc: best_acc, best_thresh = acc, thresh
                
        try: epoch_auc = roc_auc_score(all_labels_np, all_probs_np)
        except: epoch_auc = 0.0
        return epoch_loss, best_acc, epoch_auc, best_thresh

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Seed {args.seed} - Epoch {epoch}/{args.epochs}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        val_loss, val_acc, val_auc, val_thresh = evaluate(val_loader)
        scheduler.step()
        
        history["epoch"].append(epoch); history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "best_auc": best_auc, "threshold": val_thresh
            }, SAVE_MODEL_PATH)
            
    pd.DataFrame(history).to_csv(LOG_CSV_PATH, index=False)
    print(f"Log saved to {LOG_CSV_PATH}")

    return best_auc, SAVE_MODEL_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Menopause Classification Training")
    parser.add_argument('--model_name', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'vit_b_16', 'vit_b_32', 'densenet121', 'convnext_tiny', 'efficientnet_b0', 'mobilenet_v3_large'],
                        help='Name of the model to train.')
    parser.add_argument('--save_name',type=str)
    parser.add_argument("--save_root",type=str,default ="./models/1_baseline/checkpoints")
    parser.add_argument('--removal', nargs='+', default=["./data/list/Removal.csv"], help="space-separated paths to CSV files listing images to exclude")
    parser.add_argument('--data_root',type=str,default="./data/anon_data_flatten_cropped_aligned_masking")
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay for optimizer (default: 5e-3)')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of standard CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal loss (default: 2.0. Higher = more focus on hard examples)')
    parser.add_argument('--resize_mode',type=str,default='random',choices=['random','padded','zeropad','reflect'],help='padded uses replicate-border padding')
    parser.add_argument('--seed', type=int,default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    best_auc, model_path = run_training(args)
    print(f"Best single validation AUC: {best_auc:.4f} | Model saved at: {model_path}")
