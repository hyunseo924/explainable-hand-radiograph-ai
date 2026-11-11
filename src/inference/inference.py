from html import parser
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model & Analyze Probability Distribution (ECE & Histogram)")
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=['resnet18', 'resnet50', 'vit_b_16', 'vit_b_32', 'densenet121', 'convnext_tiny', 'efficientnet_b0', 'mobilenet_v3_large'], 
                        help="Model architecture (e.g., resnet18, vit_b_16)")
    parser.add_argument('--model_path', nargs='+', default=None, 
                        help="Path(s) to the trained model checkpoint(s).")
    parser.add_argument('--save_dir', type=str, default=None, 
                        help="Folder name to save result graphs (if not specified, use default path)")
    parser.add_argument('--removal', nargs='+', default=["./data/list/Removal.csv"])    
    parser.add_argument('--data_root',type=str,default= "./data/anon_data_flatten_cropped_aligned_masking")
    parser.add_argument('--resize_mode',type=str,default='random',choices=['random','padded','zeropad','reflect'],help='padded uses replicate-border padding')
    parser.add_argument('--output_csv', type=str, default='./evaluation_results.csv', help="Path to save evaluation results CSV file (if not specified, use default path)")
    parser.add_argument('--conf_csv',type=str,default="./confidence_results.csv",help="Path to save confidence results CSV file (if not specified, use default path)")
    parser.add_argument('--prediction_csv', type=str, default=None,
                        help='Path to save per-seed, per-image predictions for validation and test splits.')
    parser.add_argument('--ensemble_score_csv', type=str, default=None,
                        help='Path to save mean class-1 probabilities across evaluated checkpoints.')
    parser.add_argument("--conf_col_name",type=str,default='masked_conf')
    parser.add_argument("--region",type=str,required=True)

    return parser.parse_args()

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
    elif model_name == "densenet121":
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


SEED = 42 # dataset split seed is fixed across all runs
IMG_SIZE = 224
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def prepare_dataframe(root_dir, removal_files=None, split_seed=SEED):
    data = []
    groups = {"F_20_45": 0, "F_55_80": 1}
    
    removal_ids = set()
    if removal_files is not None:
        if isinstance(removal_files, str): removal_files = [removal_files]
        for r_file in removal_files:
            if os.path.exists(r_file):
                removal_ids.update(pd.read_csv(r_file)["id"].values)
    
    for folder_name, label in groups.items():
        folder_path = os.path.join(root_dir, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            if os.path.basename(f) in removal_ids: continue
            data.append({"image_path": f, "label": label})
            
    df = pd.DataFrame(data)
    print(f"Total samples after removal: {len(df)}")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=split_seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=split_seed)
    return train_df, val_df, test_df

class HandXrayDataset(Dataset):
    def __init__(self, dataframe, resize_mode,train=True):
        self.df = dataframe.reset_index(drop=True)
        self.train = train

        if resize_mode == "random" and self.train:
            base_preprocess = [A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.8, 1.0))]
        elif resize_mode == "padded":
            base_preprocess = [
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REPLICATE)
            ]
        elif resize_mode == "zeropad":
            base_preprocess = [
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0))
            ]
        elif resize_mode =="reflect":
            base_preprocess = [
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101) 
            ]
        else:
            base_preprocess = [A.Resize(IMG_SIZE, IMG_SIZE)]

        self.transform = A.Compose([
            #*aug_list,           
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
            img_id = os.path.splitext(os.path.basename(row["image_path"]))[0]
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return img, label, img_id
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or preprocess image: {row['image_path']}"
            ) from e

def calc_ece(probs, preds, labels, num_bins=10):
    probs, preds, labels = np.array(probs), np.array(preds), np.array(labels)
    
    confidences = np.maximum(probs, 1 - probs)
    accuracies = (preds == labels)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    for i in range(num_bins):
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lowers[i]) & (confidences <= bin_uppers[i])
        else:
            in_bin = (confidences >= bin_lowers[i]) & (confidences < bin_uppers[i])
            
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            
    return ece

def save_combined_plots(val_data, test_data, save_dir, save_name, csv_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    v_probs, v_labels = val_data['probs'], val_data['labels']
    axes[0, 0].hist(v_probs[v_labels == 0], bins=30, histtype='step', linewidth=2.5, label='GT: Class 0 (Pre-menopause)', color='dodgerblue')
    axes[0, 0].hist(v_probs[v_labels == 1], bins=30, histtype='step', linewidth=2.5, label='GT: Class 1 (Post-menopause)', color='tomato')
    axes[0, 0].hist(v_probs[v_labels == 0], bins=30, histtype='stepfilled', alpha=0.1, color='dodgerblue')
    axes[0, 0].hist(v_probs[v_labels == 1], bins=30, histtype='stepfilled', alpha=0.1, color='tomato')
    axes[0, 0].set_title('Predicted Probability Distribution (Class 1 - VAL)', fontsize=14)
    axes[0, 0].set_xlabel('Probability of Class 1', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    t_probs, t_labels = test_data['probs'], test_data['labels']
    axes[0, 1].hist(t_probs[t_labels == 0], bins=30, histtype='step', linewidth=2.5, label='GT: Class 0 (Pre-menopause)', color='dodgerblue')
    axes[0, 1].hist(t_probs[t_labels == 1], bins=30, histtype='step', linewidth=2.5, label='GT: Class 1 (Post-menopause)', color='tomato')
    axes[0, 1].hist(t_probs[t_labels == 0], bins=30, histtype='stepfilled', alpha=0.1, color='dodgerblue')
    axes[0, 1].hist(t_probs[t_labels == 1], bins=30, histtype='stepfilled', alpha=0.1, color='tomato')
    axes[0, 1].set_title('Predicted Probability Distribution (Class 1 - TEST)', fontsize=14)
    axes[0, 1].set_xlabel('Probability of Class 1', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    if os.path.exists(csv_path):
        history = pd.read_csv(csv_path)
        
        # (1,0): Loss Curve
        axes[1, 0].plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        axes[1, 0].plot(history['epoch'], history['val_loss'], label='Val Loss', color='red', linewidth=2)
        axes[1, 0].set_title('Loss Curve', fontsize=14)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # (1,1): Performance Curve
        axes[1, 1].plot(history['epoch'], history['train_acc'], label='Train Acc', color='green', linewidth=2)
        axes[1, 1].plot(history['epoch'], history['val_auc'], label='Val AUC', color='purple', linewidth=2)
        axes[1, 1].set_title('Performance Curve', fontsize=14)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, f"Log file not found:\n{os.path.basename(csv_path)}", ha='center', va='center', fontsize=12)
        axes[1, 1].text(0.5, 0.5, "Performance Data Unavailable", ha='center', va='center', fontsize=12)
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{save_name}_analysis.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    
    return save_path

def evaluate_single_model(model, loader, device, optimal_threshold):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_gt_probs = []
    all_img_ids=[]
    with torch.no_grad():
        for imgs, labels, img_ids in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]  
            preds = (probs >= optimal_threshold).float() 
            gt_probs = torch.where(labels == 1, probs, 1-probs)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gt_probs.extend(gt_probs.cpu().numpy())
            all_img_ids.extend(img_ids)
            
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ece_val = calc_ece(all_probs, all_preds, all_labels)
    per_image = pd.DataFrame({
        'image_id': all_img_ids,
        'label': all_labels,
        'prob_class1': all_probs,
        'prediction': all_preds,
        'threshold': optimal_threshold,
        'true_class_probability': all_gt_probs,
    })
    img_probs_dict = dict(zip(all_img_ids, all_gt_probs))
    return {
        'acc': acc,
        'auc': auc,
        'ece': ece_val,
        'sens': sens,
        'spec': spec,
        'img_probs': img_probs_dict,
        'per_image': per_image,
    }

if __name__ == "__main__":
    args = parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Settings] Model: {args.model_name} | Device: {DEVICE}")
    weight_files = []
    
    if len(args.model_path) == 1 and os.path.isdir(args.model_path[0]):
        pattern = os.path.join(args.model_path[0], "*.pth")
        weight_files = sorted(glob.glob(pattern))
    else:
        weight_files = [f for f in args.model_path if os.path.isfile(f)]

    if not weight_files:
        exit(1)  

    # Load data.
    _, val_df, test_df = prepare_dataframe(args.data_root, args.removal)
    val_loader = DataLoader(HandXrayDataset(val_df, args.resize_mode, train=False), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(HandXrayDataset(test_df, args.resize_mode, train=False), batch_size=32, shuffle=False, num_workers=4)

    all_results=[]
    for w_idx, w_path in enumerate(weight_files):
        print(f"\n{'='*60}")
        print(f"[{w_idx+1}/{len(weight_files)}] evaluating: {os.path.basename(w_path)}")
        
        model = get_model(args.model_name, pretrained=False).to(DEVICE)
        
        try:
            checkpoint = torch.load(w_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimal_threshold = checkpoint.get('threshold', 0.5)
                best_auc = checkpoint.get('best_auc', 'N/A')
                best_epoch = checkpoint.get('epoch', 'N/A')
                print(f"[*] Checkpoint loaded (Threshold: {optimal_threshold:.4f} | Best AUC: {best_auc} @ Epoch {best_epoch})")

            else:
                model.load_state_dict(checkpoint)
                optimal_threshold = 0.5
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {w_path}") from e
    
        val_res = evaluate_single_model(model, val_loader, DEVICE, optimal_threshold)
        test_res = evaluate_single_model(model, test_loader, DEVICE, optimal_threshold)
        
        all_results.append({
            'name': os.path.basename(w_path),
            'val': val_res,
            'test': test_res
        })
        
        print(f"   [TEST] Acc: {test_res['acc']:.4f} | AUC: {test_res['auc']:.4f} | Sens: {test_res['sens']:.4f} | Spec: {test_res['spec']:.4f}")

    if len(all_results) > 1:
        print("\n" + "#"*25)
        print(f"[{args.model_name.upper()}] summary across {len(all_results)} runs (Region: {args.region})")
        print("#"*25)
        
        metrics = ['acc', 'auc', 'ece', 'sens', 'spec']
        avg_test = {m: np.mean([r['test'][m] for r in all_results]) for m in metrics}
        std_test = {m: np.std([r['test'][m] for r in all_results]) for m in metrics}
        
        print(f"{'Metric (TEST)':<25} | {'Average':<10} | {'Std Dev':<10}")
        print("-" * 55)
        for m in metrics:
            metric_name = m.upper() if m != 'sens' and m != 'spec' else ('Sensitivity' if m == 'sens' else 'Specificity')
            print(f"{metric_name:<25} | {avg_test[m]:.4f}     | ±{std_test[m]:.4f}")
        print("="*55)

    rows = []
    metrics = ['acc', 'auc', 'ece', 'sens', 'spec']

    for r in all_results:
        row = {
            'model_name': args.model_name,
            'weight_file': f'{args.region}_{r["name"]}',
            'type': 'single',
        }
        for m in metrics:
            row[f'val_{m}']  = round(r['val'][m],  4)
            row[f'test_{m}'] = round(r['test'][m], 4)
        rows.append(row)

    if len(all_results) > 1:
        avg_row = {
            'model_name':  args.model_name,
            'weight_file': f'{args.region}_ensemble_mean ({len(all_results)} seeds)',
            'type': 'ensemble_mean',
        }
        std_row = {
            'model_name':  args.model_name,
            'weight_file': f'{args.region}_ensemble_std ({len(all_results)} seeds)',
            'type': 'ensemble_std',
        }
        avg_val  = {m: np.mean([r['val'][m]  for r in all_results]) for m in metrics}
        std_val  = {m: np.std( [r['val'][m]  for r in all_results]) for m in metrics}
        avg_test = {m: np.mean([r['test'][m] for r in all_results]) for m in metrics}
        std_test = {m: np.std( [r['test'][m] for r in all_results]) for m in metrics}

        for m in metrics:
            avg_row[f'val_{m}']  = round(avg_val[m],  4)
            avg_row[f'test_{m}'] = round(avg_test[m], 4)
            std_row[f'val_{m}']  = round(std_val[m],  4)
            std_row[f'test_{m}'] = round(std_test[m], 4)

        rows.extend([avg_row, std_row])

    result_df = pd.DataFrame(rows)
    csv_path  = args.output_csv
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    write_header = not os.path.exists(csv_path)  
    result_df.to_csv(csv_path, mode='a', header=write_header, index=False)
    print(f"\n[OK] Results saved: {csv_path}")

    if args.prediction_csv is not None and all_results:
        prediction_frames = []
        for r in all_results:
            for split_name in ('val', 'test'):
                frame = r[split_name]['per_image'].copy()
                frame.insert(0, 'split', split_name)
                frame.insert(0, 'checkpoint', r['name'])
                frame.insert(0, 'model_name', args.model_name)
                prediction_frames.append(frame)

        prediction_df = pd.concat(prediction_frames, ignore_index=True)
        prediction_path = os.path.abspath(args.prediction_csv)
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        prediction_df.to_csv(prediction_path, index=False)
        print(f"[OK] Per-seed image predictions saved: {prediction_path}")

    if args.ensemble_score_csv is not None and all_results:
        ensemble_frames = []
        for split_name in ('val', 'test'):
            per_seed_frames = []
            for r in all_results:
                frame = r[split_name]['per_image'][['image_id', 'label', 'prob_class1']].copy()
                frame = frame.rename(columns={'prob_class1': r['name']})
                per_seed_frames.append(frame)

            ensemble_df = per_seed_frames[0]
            for frame in per_seed_frames[1:]:
                ensemble_df = ensemble_df.merge(frame, on=['image_id', 'label'], how='inner', validate='one_to_one')
            score_columns = [r['name'] for r in all_results]
            ensemble_df['mean_prob_class1'] = ensemble_df[score_columns].mean(axis=1)
            ensemble_df.insert(0, 'split', split_name)
            ensemble_df.insert(0, 'model_name', args.model_name)
            ensemble_frames.append(ensemble_df)

        ensemble_score_df = pd.concat(ensemble_frames, ignore_index=True)
        ensemble_path = os.path.abspath(args.ensemble_score_csv)
        os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
        ensemble_score_df.to_csv(ensemble_path, index=False)
        print(f"[OK] Five-seed ensemble scores saved: {ensemble_path}")

    if len(all_results) > 0:
        ensemble_probs = {}
        for r in all_results:
            for img_id, prob in r['test']['img_probs'].items():
                if img_id not in ensemble_probs:
                    ensemble_probs[img_id] = []
                ensemble_probs[img_id].append(prob)
        
        avg_confs = {img_id: np.mean(probs) for img_id, probs in ensemble_probs.items()}
        new_conf_df = pd.DataFrame(list(avg_confs.items()), columns=['ImageID', args.conf_col_name])
        
        conf_csv_path = args.conf_csv
        
        if os.path.exists(conf_csv_path):
            existing_df = pd.read_csv(conf_csv_path)
            merged_df = pd.merge(existing_df, new_conf_df, on='ImageID', how='outer')
        else:
            merged_df = new_conf_df
        cols = list(merged_df.columns)
        cols.remove('ImageID') 
        origin_col = None
        for col in cols:
            if 'origin' in col.lower(): 
                origin_col = col
                break
                
        if origin_col:
            cols.remove(origin_col)
            final_cols = ['ImageID', origin_col] + cols
        else:
            final_cols = ['ImageID'] + cols
            
        merged_df = merged_df[final_cols]
        
        merged_df.to_csv(conf_csv_path, index=False)
        
        if os.path.exists(conf_csv_path):
            print(f"[OK] ImageID confidence merge completed: {conf_csv_path} (column: {args.conf_col_name})")
        else:
            print(f"[OK] ImageID confidence file created: {conf_csv_path} (column: {args.conf_col_name})")
