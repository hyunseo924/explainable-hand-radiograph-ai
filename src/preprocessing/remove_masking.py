import os
import cv2
import numpy as np
from skimage import io

input_root = "./data/anon_data_flatten_cropped_aligned"
output_root = "./data/anon_data_flatten_cropped_aligned_masking"

groups = ["F_20_45", "F_55_80"]

os.makedirs(output_root, exist_ok=True)

for group in groups:
    in_dir = os.path.join(input_root, group)
    out_dir = os.path.join(output_root, group)
    os.makedirs(out_dir, exist_ok=True)

    img_list = sorted([
        f for f in os.listdir(in_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    print(f"[INFO] Processing {group}: {len(img_list)} images")

    for fname in img_list:
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        img = img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        bin = (img > 220).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin)

        bg = cv2.medianBlur(img,151)

        remove_mask = np.zeros_like(img, dtype=np.uint8)
        inpaint_mask = np.zeros_like(img, dtype=np.uint8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if y < 0.2 * H:
                remove_mask[labels == i] = 1
            if (x < 0.2 * W) or (x + w > 0.8 * W):
                inpaint_mask[labels == i] = 1
            
        kernel_dilate = np.ones((5, 5), np.uint8)
        remove_mask = cv2.dilate(remove_mask,kernel_dilate,iterations=3)
        inpaint_mask = cv2.dilate(inpaint_mask,kernel_dilate,iterations=1)

        clean = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_TELEA)
        clean[remove_mask == 1] = bg[remove_mask == 1]

        cv2.imwrite(out_path, clean)

    print(f"[DONE] {group}")
