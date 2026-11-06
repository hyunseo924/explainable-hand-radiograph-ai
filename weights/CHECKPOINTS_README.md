# Checkpoint layout

This directory documents the expected local layout for model parameters. Trained weights are omitted from the code archive because of file-size constraints and can be downloaded from the public GitHub repository after publication. The shell scripts also work when the corresponding paths in `./configs/paths.env` point elsewhere.

```text
weights/
├── preprocessing/
│   └── laterality_resnet18_best.pth  # ResNet18 left/right classifier
├── baseline/
│   ├── resnet18/       # five best_iter5_resnet18_*_padded_seed*.pth files
│   ├── resnet50/       # five best_iter5_resnet50_*_padded_seed*.pth files
│   ├── densenet121/    # five best_iter5_densenet121_*_padded_seed*.pth files
│   ├── mobilenet_v3_large/  # five best_mobilenet_v3_large_seed*.pth files
│   └── efficientnet_b0/     # five best_efficientnet_b0_seed*.pth files
├── ablation_cnn/
│   ├── ray2_selective/ # five best_GOLD_ray2_*_padded_seed*.pth files
│   ├── ray2_selected/  # five best_GOLD_ray2_inverted_*_padded_seed*.pth files
│   ├── ray3_selective/
│   ├── ray3_selected/
│   ├── ray4_selective/
│   └── ray4_selected/
└── segmentation/
    ├── yolo/
    │   ├── metacarpal/ray{2,3,4}/checkpoints/yolov8n_bone_det/weights/best.pt
    │   └── phalanx/ray{2,3,4}/checkpoints/yolov8n_bone_det/weights/best.pt
    └── medsam/
        ├── metacarpal/crop_seg_medsam_ep25_meta{2,3,4}.pth
        └── phalanx/crop_seg_medsam_ep25_phalanx{2,3,4}.pth
```

The preprocessing checkpoint is used to orient cropped single-hand radiographs consistently before the primary analysis. The baseline and ablation CNN checkpoints contain the validation-derived decision threshold used during evaluation and inference. The YOLO and MedSAM files are component-specific segmentation parameters; each ray mask requires both its phalanx and metacarpal parameters before mask merging.

## Baseline classifiers

| Architecture | Five-file filename pattern | Resize mode |
|---|---|---|
| ResNet18 | `best_iter5_resnet18_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` | padded |
| ResNet50 | `best_iter5_resnet50_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` | padded |
| DenseNet121 | `best_iter5_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` | padded |
| MobileNetV3-Large | `best_mobilenet_v3_large_seed{42,777,1004,2024,8282}.pth` | padded |
| EfficientNet-B0 | `best_efficientnet_b0_seed{42,777,1004,2024,8282}.pth` | padded |

## DenseNet121 ray-ablation classifiers

| Condition | Five-file filename pattern |
|---|---|
| Ray 2 selective | `best_GOLD_ray2_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` |
| Ray 2 selected | `best_GOLD_ray2_inverted_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` |
| Ray 3 selective | `best_GOLD_ray3_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` |
| Ray 3 selected | `best_GOLD_ray3_inverted_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` |
| Ray 4 selective | `best_GOLD_ray4_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` |
| Ray 4 selected | `best_GOLD_ray4_inverted_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed{42,777,1004,2024,8282}.pth` |

## Segmentation components for rays 2–4

Each ray has a metacarpal and a phalanx component. The final ray mask is their pixelwise OR; the implementation is [`src/segmentation/mask_preprocessing/make_ray_mask.py`](../src/segmentation/mask_preprocessing/make_ray_mask.py).

| Component | Ray-specific parameter filename |
|---|---|
| Metacarpal YOLO | `segmentation/yolo/metacarpal/ray{2,3,4}/checkpoints/yolov8n_bone_det/weights/best.pt` |
| Phalanx YOLO | `segmentation/yolo/phalanx/ray{2,3,4}/checkpoints/yolov8n_bone_det/weights/best.pt` |
| Metacarpal MedSAM | `segmentation/medsam/metacarpal/crop_seg_medsam_ep25_meta{2,3,4}.pth` |
| Phalanx MedSAM | `segmentation/medsam/phalanx/crop_seg_medsam_ep25_phalanx{2,3,4}.pth` |

Each CNN checkpoint contains a `model_state_dict`, selected validation AUC and epoch, and a validation-derived classification threshold. Evaluation and inference load the stored threshold rather than substituting 0.5.
