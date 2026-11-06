# ======================================================
# Standard Grad-CAM  + Ensemble + Alignment Avg + PPT Subplot + Raw Data Save
# ======================================================

import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import argparse
import cv2
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--removal', nargs='+', required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet18', 'resnet50', 'vit_b_16', 'vit_b_32', 'densenet121', 'convnext_tiny', 'efficientnet_b0', 'mobilenet_v3_large'])
    parser.add_argument('--model_path', nargs='+', required=True, help="weight save path(s)")
    parser.add_argument('--save_name', type=str, required=True, help="save name for the CAM results (e.g., 'gradcam_resnet18')")
    parser.add_argument('--save_gt_only', action='store_true', help="save only when prediction is correct")
    parser.add_argument('--cam_method', type=str, default='gradcam', choices=['gradcam', 'eigencam'])
    parser.add_argument('--resize_mode', type=str, default='padded', choices=['random', 'padded', 'zeropad', 'reflect'])
    parser.add_argument('--target_class', type=str, default='pred', choices=['pred', 'origin_label', 'all'],
                        help="CAM criteria. pred(prediction), origin_label(ground truth), all(both for incorrect cases, pred for correct cases)")
    parser.add_argument('--save_raw_cam', action='store_true', help="save raw 2D Array(.npy) for further analysis")
    parser.add_argument('--save_individual_cams', action='store_true', help="save individual CAM images for each model in the ensemble")
    parser.add_argument('--save_success_only', action='store_true',
                        help="save only consensus cases whose predicted class is correct")
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--template_path', type=str, required=True)
    parser.add_argument('--overlay_alpha', type=float, default=0.50)
    parser.add_argument('--percentile_low', type=float, default=75.0)
    parser.add_argument('--percentile_high', type=float, default=99.0)
    parser.add_argument('--per_image_noise_threshold', type=float, default=0.15)
    parser.add_argument('--group_average_noise_threshold', type=float, default=0.15)
    parser.add_argument('--whole_average_noise_threshold', type=float, default=0.15)
    parser.add_argument('--dont_run_avg',action='store_true')
    return parser.parse_args()

def get_model_and_target_layer(model_name, num_classes=2):
    if model_name == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
    elif model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
    elif model_name == "densenet121":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        target_layers = [model.features[-1]]
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        target_layers = [model.features[-1][-1]]
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layers = [model.features[-1]]
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        target_layers = [model.features[-1]]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, target_layers


def load_all_models(weight_files, model_name, cam_constructor, device):
    loaded = []
    print(f"\nLoading {len(weight_files)} model(s)...")
    for w_path in weight_files:
        model_id = os.path.splitext(os.path.basename(w_path))[0]
        model, target_layers = get_model_and_target_layer(model_name)
        chkpt = torch.load(w_path, map_location=device, weights_only=False)
        model.load_state_dict(chkpt.get('model_state_dict', chkpt))
        opt_thresh = chkpt.get('threshold', 0.5) if isinstance(chkpt, dict) else 0.5
        model.to(device).eval()

        cam_extractor = cam_constructor(model=model, target_layers=target_layers)

        loaded.append({
            'model_id'   : model_id,
            'model'      : model,
            'cam'        : cam_extractor,
            'opt_thresh' : opt_thresh,
        })
        print(f"  - Loaded: {model_id}  (thresh={opt_thresh:.3f})")
    return loaded

SEED = 42
IMG_SIZE = 224
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_test_dataframe(root_dir, removal_files=None):
    data = []
    groups = {"F_20_45": 0, "F_55_80": 1}
    removal_ids = set()

    if removal_files is not None:
        if isinstance(removal_files, str):
            removal_files = [removal_files]
        for r in removal_files:
            if os.path.exists(r):
                removal_ids.update(pd.read_csv(r)["id"].values)

    for f_name, label in groups.items():
        for f in sorted(glob.glob(os.path.join(root_dir, f_name, "*.*"))):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.basename(f) not in removal_ids:
                data.append({"image_path": f, "label": label})

    df = pd.DataFrame(data)
    _, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
    return test_df

def build_inference_transform(resize_mode):
    if resize_mode == "padded":
        base_preprocess = [
            A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                          border_mode=cv2.BORDER_REPLICATE, value=None,
                          mask_value=None, position="center", p=1.0),
        ]
    elif resize_mode == "zeropad":
        base_preprocess = [
            A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                          border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                          mask_value=0, position="center", p=1.0),
        ]
    elif resize_mode == "reflect":
        base_preprocess = [
            A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                          border_mode=cv2.BORDER_REFLECT_101, value=None,
                          mask_value=None, position="center", p=1.0),
        ]
    else:
        # Evaluation uses a deterministic resize for the training mode named "random".
        base_preprocess = [A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0)]

    return A.Compose([
        *base_preprocess,
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])
import cv2
import numpy as np

def apply_inverse_shift(img, img_id, shift_dict):
    """Restore an individual CAM image using the shift recorded in a CSV file."""
    if shift_dict is None or img_id not in shift_dict:
        return img  
    
    dx = shift_dict[img_id]['dx']
    dy = shift_dict[img_id]['dy']
    
    h, w = img.shape[:2]
    
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    
    inverse_shifted_img = cv2.warpAffine(img, M, (w, h), borderValue=0)
    
    return inverse_shifted_img

def safe_filename(s: str, maxlen: int = 160) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s[:maxlen]

def save_overlay(fig, out_path):
    buf = BytesIO()
    fig.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out_path, quality=95)
    buf.close()

def get_input_geometry(img_np, resize_mode):
    """Recover the exact valid-image rectangle in the 224×224 model input.

    A binary support image is passed through the same geometric transforms as
    the model input. Constant-zero padding makes the valid rectangle observable
    without manually reproducing Albumentations rounding rules. ``random`` has
    no padding and therefore occupies the full 224×224 input.
    """
    original_h, original_w = img_np.shape[:2]
    if resize_mode == "random":
        return {
            "original_h": int(original_h), "original_w": int(original_w),
            "resized_h": IMG_SIZE, "resized_w": IMG_SIZE,
            "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0,
        }

    support = np.ones((original_h, original_w), dtype=np.uint8)
    geometry_transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_NEAREST, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                      position="center", p=1.0),
    ])
    padded_support = geometry_transform(image=support)["image"]
    if padded_support.shape != (IMG_SIZE, IMG_SIZE):
        raise RuntimeError(
            f"Unexpected padded geometry shape {padded_support.shape}; expected {(IMG_SIZE, IMG_SIZE)}"
        )
    ys, xs = np.where(padded_support > 0)
    if len(ys) == 0 or len(xs) == 0:
        raise RuntimeError("Could not determine the valid-image rectangle after padding.")

    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    return {
        "original_h": int(original_h),
        "original_w": int(original_w),
        "resized_h": int(bottom - top),
        "resized_w": int(right - left),
        "pad_top": top,
        "pad_bottom": int(IMG_SIZE - bottom),
        "pad_left": left,
        "pad_right": int(IMG_SIZE - right),
    }


def restore_cam_to_original(cam_input, geometry):
    """Undo padded-input geometry before mapping a CAM to original image space."""
    if cam_input.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"Expected an input-space CAM of {(IMG_SIZE, IMG_SIZE)}, got {cam_input.shape}. "
            "Refusing to resize an unknown coordinate system."
        )

    top = geometry["pad_top"]
    left = geometry["pad_left"]
    bottom = top + geometry["resized_h"]
    right = left + geometry["resized_w"]
    cam_unpadded = cam_input[top:bottom, left:right]
    if cam_unpadded.shape != (geometry["resized_h"], geometry["resized_w"]):
        raise RuntimeError("CAM crop does not match the recorded valid-image rectangle.")

    # Remove model-input padding before PIL uint8 bilinear resizing.
    cam_img = Image.fromarray((cam_unpadded * 255).astype(np.uint8)).resize(
        (geometry["original_w"], geometry["original_h"]), Image.BILINEAR
    )
    return np.array(cam_img).astype(np.float32) / 255.0


def render_cam_overlay(img_rgb, cam_original, alpha, p_low_pct, p_high_pct, noise_thresh):
    """Post-process a CAM already restored to the original image coordinate system."""
    expected_shape = (img_rgb.height, img_rgb.width)
    if cam_original.shape != expected_shape:
        raise ValueError(
            f"CAM/original-image shape mismatch: CAM={cam_original.shape}, image={expected_shape}."
        )
    cam_rs = cam_original.astype(np.float32, copy=True)
    p_low  = np.percentile(cam_rs, p_low_pct)
    p_high = np.percentile(cam_rs, p_high_pct)
    
    cam_rs = np.clip(cam_rs, p_low, p_high)
    cam_rs = (cam_rs - p_low) / (p_high - p_low + 1e-8)

    cam_rs[cam_rs < noise_thresh] = 0.0 

    cam_rs_vis = np.where(cam_rs == 0.0, np.nan, cam_rs)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img_rgb, cmap="gray")
    plt.imshow(cam_rs_vis, cmap="turbo", alpha=alpha)
    plt.axis("off")
    
    return fig, cam_rs

def generate_simple_average(cam_dir, template_path, output_dir, target_gt, target_pred, cam_method, img_dict,
                            alpha=0.50, noise_thresh=0.15):
    group_name = {0: "younger adult group", 1: "older adult group"}.get(target_gt, f"GT{target_gt}")
    # Load the template image.
    if not os.path.exists(template_path):
        print(f"template path is invalid : {template_path}")
        return None, 0
    
    template_basename=os.path.basename(template_path)
    template_origin_path=img_dict[os.path.splitext(template_basename)[0]]
    if not template_origin_path:
        print(f"template origin path is invalid : {template_origin_path}")
        return None, 0
    
    bg_sharp = cv2.imread(template_origin_path)
    if bg_sharp is None:
        print(f"template origin image could not be loaded : {template_origin_path}")
        return None, 0
    target_gray = cv2.cvtColor(bg_sharp, cv2.COLOR_BGR2GRAY)
    h, w = target_gray.shape

    # find raw cam data(.npy)
    parent_dir=os.path.dirname(os.path.abspath(cam_dir))
    raw_dir = os.path.join(parent_dir, "raw_cam_data")
    all_files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))
    
    filtered_files = [f for f in all_files 
                      if f"GT{target_gt}" in os.path.basename(f) 
                      and f"pred{target_pred}" in os.path.basename(f)]

    if not filtered_files:
        print(f"Filtered Raw Data files not found: GT{target_gt}, Pred{target_pred}")
        return None, 0

    sum_cam_raw = np.zeros((h, w), dtype=np.float32)
    succ_count, fail_count = 0, 0
    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    warp_mode = cv2.MOTION_AFFINE
    
    pbar = tqdm(filtered_files, desc=f"Avg (GT{target_gt}, Pred{target_pred})")
    
    for path in pbar:
        cam_raw = np.load(path)
        cam_raw = np.nan_to_num(cam_raw,nan=0.0)
        basename = os.path.basename(path)
        match = re.search(r'PID(.*?)_GT', basename)
        pid = match.group(1) if match else None
        
        if not pid or pid not in img_dict:
            fail_count += 1
            pbar.set_postfix(succ=succ_count, fail=fail_count)
            continue

        orig_img_path = img_dict[pid]
        orig_img = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            fail_count += 1
            continue
            
        if orig_img.shape[:2] != (h, w): orig_img = cv2.resize(orig_img, (w, h))
        if cam_raw.shape != (h, w): cam_raw = cv2.resize(cam_raw, (w, h))
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(target_gray, orig_img, warp_matrix, warp_mode, criteria)
            
            aligned_cam_raw = cv2.warpAffine(cam_raw, warp_matrix, (w, h), 
                                             flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)
            
            sum_cam_raw += aligned_cam_raw
            succ_count += 1
            pbar.set_postfix(succ=succ_count, fail=fail_count)
        except cv2.error:
            fail_count += 1
            pbar.set_postfix(succ=succ_count, fail=fail_count)

    if succ_count == 0: 
        print(f"GT{target_gt}_pred{target_pred}: all registrations failed")
        return None, 0
    
    avg_cam_raw = sum_cam_raw / succ_count    
    avg_cam_raw[avg_cam_raw < noise_thresh] = 0.0
    
    heat_rgba = plt.cm.turbo(avg_cam_raw)
    heat_rgb  = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
    heat_bgr  = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    
    masked_bg = cv2.imread(template_path)
    if len(masked_bg.shape)==2:
        bg_gray3 = cv2.cvtColor(masked_bg, cv2.COLOR_GRAY2BGR)
    else:
        bg_gray3 = masked_bg.copy()

    final_bgr = bg_gray3.copy()
    mask = avg_cam_raw > 0.0
    roi = cv2.addWeighted(bg_gray3, 1 - alpha, heat_bgr, alpha, 0)    
    final_bgr[mask] = roi[mask]

    # save
    output_file = os.path.join(
        output_dir,
        f"Avg_{cam_method.upper()}_GT{target_gt}_Pred{target_pred}_{succ_count}of{succ_count + fail_count}.png"
    )
    cv2.imwrite(output_file, final_bgr)
    print(
        f"[Average inclusion] {group_name} (GT{target_gt}/Pred{target_pred}): "
        f"{succ_count}/{len(filtered_files)} patient-level CAMs successfully registered "
        f"({fail_count} registration failures)."
    )

    return sum_cam_raw, succ_count

def generate_whole_average(success_bundles,avg_dir,img_dict,template_path,cam_method,
                           noise_thres=0.15, alpha=0.50):
    print("\n[Whole Average] Generating whole average...")
    
    template_basename = os.path.basename(template_path)
    template_pid = os.path.splitext(template_basename)[0]
    template_origin_path = img_dict[template_pid]
    
    bg_sharp = cv2.imread(template_origin_path)
    target_gray = cv2.cvtColor(bg_sharp, cv2.COLOR_BGR2GRAY)
    h, w = target_gray.shape
    
    whole_sum_raw = np.zeros((h, w), dtype=np.float32)
    whole_total_count = 0
    
    for sum_arr, succ_cnt in success_bundles:
        whole_sum_raw += sum_arr
        whole_total_count += succ_cnt
        
    avg_whole_raw = whole_sum_raw / whole_total_count
    
    avg_whole_raw[avg_whole_raw < noise_thres] = 0.0
    
    heat_rgba = plt.cm.turbo(avg_whole_raw)
    heat_rgb  = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
    heat_bgr  = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    
    masked_bg = cv2.imread(template_path)
    if len(masked_bg.shape) == 2:
        bg_gray3 = cv2.cvtColor(masked_bg, cv2.COLOR_GRAY2BGR)
    else:
        bg_gray3 = masked_bg.copy()
        
    final_bgr = bg_gray3.copy()
    mask = avg_whole_raw > 0.0
    
    roi = cv2.addWeighted(bg_gray3, 1 - alpha, heat_bgr, alpha, 0) # alpha=0.5
    final_bgr[mask] = roi[mask]

    whole_output_file = os.path.join(
        avg_dir, 
        f"Avg_{cam_method.upper()}_WHOLE_SUCCESS_{whole_total_count}samples.png"
    )
    cv2.imwrite(whole_output_file, final_bgr)
    print(f"[Whole-average inclusion] {whole_total_count} successfully registered patient-level CAMs.")
    print(f"Whole Average save completed: {whole_output_file}")

def save_combined_averages(avg_dir, root_all_dir, save_name, cam_method, ensemble_str):
    avg_files = sorted(glob.glob(os.path.join(avg_dir, f"Avg_{cam_method.upper()}_*.png")))
    if not avg_files: return

    num_imgs = len(avg_files)
    fig, axes = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5))
    if num_imgs == 1: axes = [axes]

    for ax, img_path in zip(axes, avg_files):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis('off')
        basename   = os.path.basename(img_path)
        gt_match   = re.search(r'GT(\d+)',   basename)
        pred_match = re.search(r'Pred(\d+)', basename)
        title = (f"Ground Truth: {gt_match.group(1)} | Prediction: {pred_match.group(1)}"
                 if gt_match and pred_match else basename.split('_')[0])
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    plt.tight_layout()
    combined_save_path = os.path.join(root_all_dir, f"Combined_Averages_{save_name}{ensemble_str}.jpg")
    fig.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"PPT subplot image saved at\n    {combined_save_path}")


if __name__ == "__main__":
    args   = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")
    if not os.path.isfile(args.template_path):
        raise FileNotFoundError(f"template_path does not exist: {args.template_path}")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("overlay_alpha must be between 0 and 1.")
    if not (0.0 <= args.percentile_low < args.percentile_high <= 100.0):
        raise ValueError("Require 0 <= percentile_low < percentile_high <= 100.")
    for threshold_name, threshold_value in (
        ("per_image_noise_threshold", args.per_image_noise_threshold),
        ("group_average_noise_threshold", args.group_average_noise_threshold),
        ("whole_average_noise_threshold", args.whole_average_noise_threshold),
    ):
        if not (0.0 <= threshold_value <= 1.0):
            raise ValueError(f"{threshold_name} must be between 0 and 1.")

    weight_files = []
    if len(args.model_path) == 1 and os.path.isdir(args.model_path[0]):
        weight_files = glob.glob(os.path.join(args.model_path[0], f"*{args.model_name}*.pth"))
    else:
        weight_files = [f for f in args.model_path if os.path.isfile(f)]
    weight_files = sorted(weight_files)

    num_models = len(weight_files)
    if num_models == 0:
        print("[ERROR] No valid checkpoint files were found.")
        exit(1)

    print(f"\n[Settings] Model: {args.model_name} | Target Mode: {args.target_class.upper()}")
    print(f"[*] Generating {'ensemble' if num_models > 1 else 'single-model'} CAMs from {num_models} checkpoint(s).")
    print(
        "[*] CAM settings: "
        f"per-image p{args.percentile_low:g}–p{args.percentile_high:g}, "
        f"per-image threshold={args.per_image_noise_threshold:g}, "
        f"group threshold={args.group_average_noise_threshold:g}, "
        f"whole threshold={args.whole_average_noise_threshold:g}, "
        f"alpha={args.overlay_alpha:g}."
    )

    base_save_dir = os.path.join(args.save_dir, f"gradcam_results_{args.model_name}")
    ensemble_str = f"_{num_models}models" if num_models > 1 else ""

    ALL_DIR = os.path.join(base_save_dir, f"all_{args.cam_method}_{args.save_name}{ensemble_str}")
    os.makedirs(ALL_DIR, exist_ok=True)

    OUT_DIR = os.path.join(ALL_DIR, f"{args.cam_method}_result_{args.save_name}{ensemble_str}")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.save_raw_cam:
        RAW_DIR = os.path.join(ALL_DIR, "raw_cam_data")
        os.makedirs(RAW_DIR, exist_ok=True)

    if args.save_individual_cams and num_models > 1:
        INDIVIDUAL_DIR = os.path.join(OUT_DIR, "individual_models")
        os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

    cam_constructor = GradCAM if args.cam_method == 'gradcam' else EigenCAM
    all_model_info  = load_all_models(weight_files, args.model_name, cam_constructor, DEVICE)

    test_df = get_test_dataframe(args.data_root, args.removal)

    ALPHA = args.overlay_alpha
    P_LOW = args.percentile_low
    P_HIGH = args.percentile_high
    PER_IMAGE_NOISE_THRESH = args.per_image_noise_threshold
    GROUP_AVERAGE_NOISE_THRESH = args.group_average_noise_threshold
    WHOLE_AVERAGE_NOISE_THRESH = args.whole_average_noise_threshold
    tf_visual = build_inference_transform(args.resize_mode)
    geometry_records = []

    run_settings = {
        "coordinate_restoration": "crop model-input padding before resizing CAM to original image dimensions",
        "model_input_size": [IMG_SIZE, IMG_SIZE],
        "resize_mode": args.resize_mode,
        "padding_position": "center",
        "padding_border_mode": "replicate" if args.resize_mode == "padded" else args.resize_mode,
        "cam_interpolation": "PIL.Image.BILINEAR",
        "overlay_alpha": ALPHA,
        "percentile_low": P_LOW,
        "percentile_high": P_HIGH,
        "per_image_noise_threshold": PER_IMAGE_NOISE_THRESH,
        "group_average_noise_threshold": GROUP_AVERAGE_NOISE_THRESH,
        "whole_average_noise_threshold": WHOLE_AVERAGE_NOISE_THRESH,
        "template_path": os.path.abspath(args.template_path),
        "checkpoint_paths": [os.path.abspath(path) for path in weight_files],
    }
    with open(os.path.join(ALL_DIR, "gradcam_run_settings.json"), "w", encoding="utf-8") as handle:
        json.dump(run_settings, handle, indent=2)

    skipped_no_consensus = 0
    skipped_gt_only      = 0

    print(f"\nGenerating CAMs and saving to: {ALL_DIR}")

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="CAM Generation"):
        img_path = row["image_path"]
        gt       = int(row["label"])
        pid      = os.path.splitext(os.path.basename(img_path))[0]

        img_rgb = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_rgb)
        geometry = get_input_geometry(img_np, args.resize_mode)
        geometry_records.append({"image_id": pid, "ground_truth": gt, **geometry})
        x       = tf_visual(image=img_np)['image'].unsqueeze(0).to(DEVICE)

        per_model_results = []

        for info in all_model_info:
            model      = info['model']
            cam_ext    = info['cam']
            opt_thresh = info['opt_thresh']
            model_id   = info['model_id']

            with torch.no_grad():
                prob = torch.softmax(model(x)[0], dim=0).cpu().numpy()[1]

            model_pred = 1 if prob >= opt_thresh else 0

            targets_to_extract = []
            if args.target_class == 'pred':
                targets_to_extract.append(('pred', model_pred))
            elif args.target_class == 'origin_label':
                targets_to_extract.append(('origin_label', gt))
            elif args.target_class == 'all':
                targets_to_extract.append(('pred', model_pred))
                if model_pred != gt:
                    targets_to_extract.append(('origin_label', gt))

            model_cams = {}
            for key, cls_idx in targets_to_extract:
                grayscale_cam = cam_ext(input_tensor=x, targets=[ClassifierOutputTarget(cls_idx)])[0, :]
                cam_min, cam_max = grayscale_cam.min(), grayscale_cam.max()
                norm_cam = (grayscale_cam - cam_min) / (cam_max - cam_min + 1e-7)
                model_cams[key] = restore_cam_to_original(norm_cam, geometry)

            per_model_results.append({
                'model_id' : model_id,
                'pred'     : model_pred,
                'prob'     : float(prob),
                'cams'     : model_cams,
            })

        all_preds     = [r['pred'] for r in per_model_results]
        has_consensus = (len(set(all_preds)) == 1)

        if not has_consensus:
            skipped_no_consensus += 1
            if args.save_individual_cams and num_models > 1:
                for r in per_model_results:
                    for key, cam_raw in r['cams'].items():
                        fig, _ = render_cam_overlay(img_rgb, cam_raw, ALPHA, P_LOW, P_HIGH, PER_IMAGE_NOISE_THRESH)
                        ind_filename = (f"{i:04d}_PID{pid}_GT{gt}_pred{r['pred']}"
                                        f"_P1_{r['prob']:.3f}_target_{key}"
                                        f"_NO_CONSENSUS_{r['model_id']}.jpg")
                        save_overlay(fig, os.path.join(INDIVIDUAL_DIR, safe_filename(ind_filename)))
            continue  
        consensus_pred = all_preds[0]
        avg_p1         = float(np.mean([r['prob'] for r in per_model_results]))

        if (args.save_gt_only or args.save_success_only) and (consensus_pred != gt):
            skipped_gt_only += 1
            if args.save_individual_cams and num_models > 1:
                for r in per_model_results:
                    for key, cam_raw in r['cams'].items():
                        fig, _ = render_cam_overlay(img_rgb, cam_raw, ALPHA, P_LOW, P_HIGH, PER_IMAGE_NOISE_THRESH)
                        ind_filename = (f"{i:04d}_PID{pid}_GT{gt}_pred{r['pred']}"
                                        f"_P1_{r['prob']:.3f}_target_{key}"
                                        f"_{r['model_id']}.jpg")
                        save_overlay(fig, os.path.join(INDIVIDUAL_DIR, safe_filename(ind_filename)))
            continue

        ensemble_cams = {}
        for r in per_model_results:
            for key, cam_raw in r['cams'].items():
                ensemble_cams.setdefault(key, []).append(cam_raw)

        for key, cam_list in ensemble_cams.items():
            avg_cam = np.mean(cam_list, axis=0)
            fig, cam_rs = render_cam_overlay(img_rgb, avg_cam, ALPHA, P_LOW, P_HIGH, PER_IMAGE_NOISE_THRESH)

            base_filename = (f"{i:04d}_PID{pid}_GT{gt}_pred{consensus_pred}"
                             f"_P1_{avg_p1:.3f}_target_{key}"
                             f"_{args.cam_method.upper()}_ENSEMBLE")

            save_overlay(fig, os.path.join(OUT_DIR, safe_filename(base_filename + ".jpg")))

            if args.save_raw_cam:
                np.save(os.path.join(RAW_DIR, safe_filename(base_filename + ".npy")), cam_rs)

        if args.save_individual_cams and num_models > 1:
            for r in per_model_results:
                for key, cam_raw in r['cams'].items():
                    fig, _ = render_cam_overlay(img_rgb, cam_raw, ALPHA, P_LOW, P_HIGH, PER_IMAGE_NOISE_THRESH)
                    ind_filename = (f"{i:04d}_PID{pid}_GT{gt}_pred{r['pred']}"
                                    f"_P1_{r['prob']:.3f}_target_{key}"
                                    f"_{r['model_id']}.jpg")
                    save_overlay(fig, os.path.join(INDIVIDUAL_DIR, safe_filename(ind_filename)))

    print(f"  - # of Consensus mismatch : {skipped_no_consensus}")
    if args.save_gt_only or args.save_success_only:
        print(f"  - # of samples skipped due to success-only filtering : {skipped_gt_only}")
    pd.DataFrame(geometry_records).to_csv(os.path.join(ALL_DIR, "cam_input_geometry.csv"), index=False)
    
    if args.dont_run_avg:
        print("\n skip averaging due to --dont_run_avg")
        print(f"individual CAM results can be found in {OUT_DIR}")
        exit(0)

    print("\n generating average CAMs...")
    img_dict ={}
    for root,_,files in os.walk(args.data_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                pid = os.path.splitext(f)[0]
                img_dict[pid] = os.path.join(root,f)

    TEMPLATE_PATH = args.template_path
    AVG_OUT_DIR   = os.path.join(ALL_DIR, f'avg_{args.cam_method}_{args.save_name}{ensemble_str}')
    os.makedirs(AVG_OUT_DIR, exist_ok=True)
    success_bundles=[]
    for TARGET_GT in (0, 1):
        for TARGET_PRED in (0, 1):
            if args.save_success_only and (TARGET_GT != TARGET_PRED):
                continue
            sum_cam_raw, succ_count = generate_simple_average(
                OUT_DIR, TEMPLATE_PATH, AVG_OUT_DIR, TARGET_GT, TARGET_PRED, args.cam_method, img_dict,
                alpha=ALPHA, noise_thresh=GROUP_AVERAGE_NOISE_THRESH,
            )
            if TARGET_GT == TARGET_PRED and succ_count > 0:
                success_bundles.append((sum_cam_raw, succ_count))

    if success_bundles:
        generate_whole_average(
            success_bundles, AVG_OUT_DIR, img_dict, TEMPLATE_PATH, args.cam_method,
            noise_thres=WHOLE_AVERAGE_NOISE_THRESH, alpha=ALPHA,
        )

    save_combined_averages(AVG_OUT_DIR, ALL_DIR, args.save_name, args.cam_method, ensemble_str)

    print("\n all processes completed.")
    print(f"Results can be found in: {ALL_DIR}")
