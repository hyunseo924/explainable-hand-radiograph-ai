import os
import glob
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="utility to apply morphology to existing masks and merge with original images at lightning speed")
    parser.add_argument('--img_dir', type=str, default="./data/anon_data_flatten_cropped_aligned_masking", help="original X-ray directory")
    parser.add_argument('--mask_dir', type=str, required=True, help="existing extracted masks (pseudo_masks) directory")
    parser.add_argument('--save_dir', type=str, required=True, help="top-level directory where results will be saved")
    parser.add_argument('--save_name', type=str, required=True, help="save folder name (e.g., refine_meta12)")

    parser.add_argument('--morph_size', type=int, default=5, help="morphology (Close->Open) kernel size. (0 means use original mask as is)")
    parser.add_argument('--invert', action='store_true', 
                        help="[enabled] keep only bone (mask) parts / [disabled] erase bone (mask) parts and keep background")
    parser.add_argument('--dont_save_mask', action='store_true', help="whether to save mask files")
    parser.add_argument('--contour', action='store_true', help="fill all holes inside the contour")
    parser.add_argument('--blur_fill',action='store_true',help="fill background with blur when removing mask")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.dont_save_mask:
        out_mask_dir = os.path.join(args.save_dir, f"refined_masks_{args.save_name}")
    out_img_dir = os.path.join(args.save_dir, f"masked_images_{args.save_name}")

    if args.morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph_size, args.morph_size))
        print(f"Morphology operation enabled (kernel size: {args.morph_size}x{args.morph_size})")
    else:
        print("Morphology operation disabled (using original masks)")

    print(f"Invert option: {'Enabled (keeping only mask parts)' if args.invert else 'Disabled (erasing mask parts and keeping background)'}")

    print("Scanning original image files and creating mapping...")
    img_dict = {}
    for root, _, files in os.walk(args.img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                basename_no_ext = os.path.splitext(file)[0]
                img_dict[basename_no_ext] = os.path.join(root, file)

    print(f"found total {len(img_dict)} original images.")

    # Scan mask files.
    mask_files = []
    for root, _, files in os.walk(args.mask_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_files.append(os.path.join(root, file))

    if not mask_files:
        print(f"[!] No masks found. Please check the path: {args.mask_dir}")
        exit(1)

    print(f"\nfound total {len(mask_files)} masks to process.")

    success_cnt = 0
    fail_cnt = 0

    for mask_path in tqdm(mask_files, desc="Processing"):
        mask_filename = os.path.basename(mask_path)
        mask_basename_no_ext = os.path.splitext(mask_filename)[0]        
        if mask_basename_no_ext not in img_dict:
            fail_cnt += 1
            continue
            
        img_path = img_dict[mask_basename_no_ext]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            fail_cnt += 1
            continue
        if args.morph_size > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if args.contour:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
            mask = contour_mask
            
      
        if args.blur_fill:
            h, w = img.shape[:2]
            scale_down = max(1, w // 20)
            scale_h = max(1, h // 20)
            small_img = cv2.resize(img, (scale_down, scale_h), interpolation=cv2.INTER_LINEAR)
            blurred_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)
            blurred_img = cv2.GaussianBlur(blurred_img, (99, 99), 0)

            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_bool = mask_3ch > 0 
            final_img = np.where(mask_bool, blurred_img, img)

        else:
            if args.invert:
                applied_mask = mask
            else:
                applied_mask = cv2.bitwise_not(mask)

            mask_3ch = cv2.cvtColor(applied_mask, cv2.COLOR_GRAY2RGB)
            final_img = cv2.bitwise_and(img, mask_3ch)

        clean_rel_path = os.path.relpath(img_path, args.img_dir)
        save_i_path = os.path.join(out_img_dir, clean_rel_path)
        
        os.makedirs(os.path.dirname(save_i_path), exist_ok=True)
        cv2.imwrite(save_i_path, final_img)

        if not args.dont_save_mask:
            save_m_path = os.path.join(out_mask_dir, clean_rel_path)
            os.makedirs(os.path.dirname(save_m_path), exist_ok=True)
            save_m_path = os.path.splitext(save_m_path)[0] + ".png"
            cv2.imwrite(save_m_path, mask)
        
        success_cnt += 1
