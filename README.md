# Explainable deep learning localizes age-discriminative candidate regions on hand radiographs of adult women

*Supplementary Code 2: Model Development and Image Analysis Workflow*

## Overview

This package contains source code for baseline classification, Grad-CAM analysis, ray-based masking experiments, bootstrap confidence-interval estimation, and sensitivity analyses of checkpoint selection and data partitioning. The custom quantitative measurement tool is provided separately as Supplementary Code 1.

This archive contains source code, runnable workflow scripts, and a checkpoint-layout guide. Trained model weights are omitted because of file-size constraints; after publication, they can be downloaded from the public GitHub repository and placed under `./weights/` as described in [`weights/CHECKPOINTS_README.md`](./weights/CHECKPOINTS_README.md).

The original radiographs and patient-level data are not included in this archive and will not be deposited in the public repository.

## Repository structure

```text
Supplementary_Code_2/
├── README.md
├── requirements.txt
├── configs/
│   └── example_paths.env                 # copy to paths.env and set local paths
├── scripts/                              # executable workflow scripts, in run order
│   ├── 00_export_fixed_split_manifest.sh
│   ├── 01_baseline_train.sh
│   ├── 02_baseline_evaluate.sh
│   ├── 03_baseline_gradcam.sh
│   ├── 04_train_yolo_components.sh
│   ├── 05_train_medsam_components.sh
│   ├── 06a_infer_component_masks.sh
│   ├── 06b_merge_reviewed_masks_and_apply.sh
│   ├── 07_ablation_train.sh
│   ├── 08_ablation_evaluate.sh
│   ├── 09_ablation_gradcam.sh
│   ├── 10_revision_kfold.sh
│   ├── 11_revision_one_se.sh
│   └── 12_revision_seed_mean_bootstrap_ci.sh
├── src/
│   ├── preprocessing/                    # laterality alignment and artifact removal
│   │   └── RL_alignment/                 # train, predict, and horizontal alignment
│   ├── train/                            # five-seed classifier training
│   ├── inference/                        # fixed-split and fold-level model evaluation
│   ├── analysis/                         # Grad-CAM generation, registration, and averaging
│   └── segmentation/                     # YOLO/MedSAM processing and ray-mask utilities
├── analysis/
│   ├── kfold/                            # seven-fold manifest, training, evaluation, and Table S5 summary
│   ├── one_se/                           # adapted 1SE checkpoint selection
│   └── bootstrap_ci/                     # five-seed mean bootstrap confidence intervals
└── weights/                              # checkpoint placement guide; no parameter files
    └── CHECKPOINTS_README.md              # expected filenames and directory layout
```

The `weights/` directory contains only [`weights/CHECKPOINTS_README.md`](./weights/CHECKPOINTS_README.md), which documents where downloaded checkpoints should be placed; trained parameter files are not included.

## Software environment

The analyses used the conda environment `bone_env` with Python 3.10.19, PyTorch 2.6.0, and torchvision 0.21.0. Package versions from that environment are pinned in [`requirements.txt`](./requirements.txt). Install a CUDA-compatible PyTorch build appropriate to the local system, then install the remaining packages.

```bash
conda create -n bone_env python=3.10.19
conda activate bone_env
pip install -r ./requirements.txt
```

`opencv-python` and `opencv-python-headless` both appeared in the recorded environment; users may retain the compatible OpenCV package for their platform.

## Input data layout

The classification and Grad-CAM workflows expect one de-identified PA hand radiograph per participant:

```text
DATA_ROOT/
├── F_20_45/    # class 0: younger adult group
└── F_55_80/    # class 1: older adult group
```

JPEG and PNG images are supported. `Removal.csv` must contain an `id` column when images are to be excluded; each value must match an image filename. The cohort must contain no more than one radiograph per participant.

Before classification, single-hand radiographs are oriented consistently using the laterality workflow in `src/preprocessing/RL_alignment/`. `train_RL.py` trains the ResNet18 left/right classifier, `predict_RL.py` writes its prediction and confidence for each image, and `apply_align.py` horizontally flips images labelled as left hands. Predictions below the fixed confidence threshold of 0.85 are left blank in the `final` column for verification before alignment.

For segmentation, the manually labelled component masks are expected beneath `SEGMENTATION_LABEL_ROOT`:

```text
SEGMENTATION_LABEL_ROOT/
├── Metacarpal/Metacarpal_{2,3,4}/
│   ├── masks/
│   └── list/masks.csv
└── Phalanx/Phalanx_{2,3,4}/
    ├── masks/
    └── list/masks.csv
```

Each `masks.csv` contains `image_id`. Script `06a_infer_component_masks.sh` writes the automatically generated component masks to the following directories:

```text
SEGMENTATION_OUTPUT_ROOT/inference/
├── metacarpal/rayN/metacarpal_rayN/pseduo_masks_metacarpal_rayN/{F_20_45,F_55_80}/
└── phalanx/rayN/phalanx_rayN/pseduo_masks_phalanx_rayN/{F_20_45,F_55_80}/
```

These automatic outputs must be visually reviewed and manually corrected where necessary before ray-mask construction. Store the reviewed component masks in the following layout while retaining the class subdirectories and image filenames:

```text
REFINED_COMPONENT_MASK_ROOT/
├── metacarpal/ray{2,3,4}/{F_20_45,F_55_80}/
└── phalanx/ray{2,3,4}/{F_20_45,F_55_80}/
```

Ray *r* is produced only from these reviewed masks by combining its metacarpal and phalanx components with a pixelwise OR operation.

## Local paths

The scripts need to know where the local image data, removal list, checkpoint files, and Grad-CAM registration template are stored. These locations differ by system, so they are kept in one small local configuration file instead of being written into the source code. Create it once from the example and replace only the placeholder paths:

```bash
cp ./configs/example_paths.env ./configs/paths.env
```

Every executable script below reads `./configs/paths.env` and runs from the package root. Scripts use GPU 0 by default (`CUDA_VISIBLE_DEVICES=0`); set the environment variable before execution to choose another visible device. `TEMPLATE_RELATIVE_PATH` is the image path of the fixed Grad-CAM registration reference relative to each corresponding image root.

## Primary training settings

| Item | Setting |
|---|---|
| Fixed split | sorted filenames; stratified 70%/15%/15% training/validation/test split; split seed 42 |
| Training seeds | 42, 777, 2024, 1004, 8282 |
| Architectures | ResNet18, ResNet50, DenseNet121, MobileNetV3-Large, EfficientNet-B0 |
| Initialization and loss | ImageNet-1K pretrained initialization; cross-entropy loss |
| Optimizer | AdamW; learning rate 2e-5; weight decay 0.005; cosine annealing to 1e-8 |
| Training | 30 epochs; batch size 32; image size 224; no early stopping |
| Reported resize mode | longest-side resize to 224 followed by replicate padding (`padded`) |
| Normalization | fixed ImageNet mean `(0.485, 0.456, 0.406)` and SD `(0.229, 0.224, 0.225)` |
| Training augmentation | rotation ±7°, brightness/contrast jitter, CLAHE, horizontal flipping |
| Checkpoint selection | highest validation AUC; each checkpoint stores the threshold maximizing validation accuracy over 0.10–0.90 in increments of 0.01 |

The ImageNet normalization constants are fixed pretrained-model constants; they are not estimated from this study dataset. The classification threshold stored with each selected checkpoint is held fixed during test evaluation and external inference.

## Run order and executable scripts

Run commands from the package root, for example `bash ./scripts/01_baseline_train.sh`. Each script uses the final padded preprocessing, fixed split seed, five training seeds, and hyperparameters listed above unless it is explicitly a revision sensitivity analysis.

If laterality preprocessing is required for a new cohort, place cropped single-hand images under `./data/anon_data_flatten_cropped`. The checkpoint used in the reported preprocessing is named `laterality_resnet18_best.pth`; when obtained from the public GitHub repository, place it at `./weights/preprocessing/laterality_resnet18_best.pth`. It can alternatively be retrained with the first command below from labelled images arranged under `./data/pseudo_laterality/{left,right}/`. Run the commands from the package root. `predict_RL.py` writes `./data/predictions_add.csv`; predictions with confidence below 0.85 have a blank `final` field. Review those cases and complete the verified laterality in the `final` column before running `apply_align.py`. The alignment step saves consistently oriented radiographs under `./data/anon_data_flatten_cropped_aligned` by horizontally flipping images labelled as left hands.

```bash
python ./src/preprocessing/RL_alignment/train_RL.py
python ./src/preprocessing/RL_alignment/predict_RL.py
python ./src/preprocessing/RL_alignment/apply_align.py
```

| Order | Script | Purpose and manuscript output |
|---|---|---|
| 0 | `./scripts/00_export_fixed_split_manifest.sh` | Writes the sorted, seed-42 fixed-split manifest used to audit the primary train/validation/test allocation. |
| 1 | `./scripts/01_baseline_train.sh` | Trains the five baseline architectures with five random seeds each. |
| 2 | `./scripts/02_baseline_evaluate.sh` | Evaluates five checkpoints per baseline and writes seed-level, per-image, and score CSVs used for the primary baseline table. |
| 3 | `./scripts/03_baseline_gradcam.sh` | Generates individual and raw Grad-CAM maps and the class-specific and whole successful-case averages for the baseline Grad-CAM figures. |
| 4 | `./scripts/04_train_yolo_components.sh` | Trains component-specific YOLO cropping models for metacarpal and phalanx regions of rays 2–4. The script passes component-specific class names such as `metacarpal_ray2` and `phalanx_ray2` to `YOLO.py`. |
| 5 | `./scripts/05_train_medsam_components.sh` | Fine-tunes component-specific MedSAM segmentation models for rays 2–4. |
| 6a | `./scripts/06a_infer_component_masks.sh` | Performs YOLO/MedSAM inference separately for the metacarpal and phalanx components of rays 2–4. Stop after this script to visually review the masks and manually correct them where necessary. |
| 6b | `./scripts/06b_merge_reviewed_masks_and_apply.sh` | Reads only the reviewed component masks from `REFINED_COMPONENT_MASK_ROOT`, merges the matched metacarpal and phalanx masks, and creates the selective and selected ray image sets. |
| 7 | `./scripts/07_ablation_train.sh` | Trains five DenseNet121 seeds for each selective and selected ray 2–4 condition. |
| 8 | `./scripts/08_ablation_evaluate.sh` | Evaluates the six ray-ablation conditions and writes the CSVs used for the ablation table and CI analysis. |
| 9 | `./scripts/09_ablation_gradcam.sh` | Generates Grad-CAM outputs for the six ray-ablation conditions. |
| 10 | `./scripts/10_revision_kfold.sh` | Generates seven-fold manifests, performs the baseline and ray-ablation sensitivity analyses, and writes the fold-level and Supplementary Table S5 summaries. |
| 11 | `./scripts/11_revision_one_se.sh` | Performs the Max-AUC versus adapted 1SE checkpoint-selection sensitivity analysis and test evaluation. |
| 12 | `./scripts/12_revision_seed_mean_bootstrap_ci.sh` | Calculates the 20,000-replicate patient-level stratified bootstrap CIs for the five-seed mean metrics in Tables 2 and 3. |

The main-text and supplementary linkage is: primary baseline performance (scripts 1–2), Grad-CAM figures (script 3), ray selection/ablation performance and images (scripts 4–9), seven-fold analysis (script 10), checkpoint-selection sensitivity analysis (script 11), and five-seed mean confidence intervals (script 12).

## Grad-CAM analysis

`./scripts/03_baseline_gradcam.sh` and `./scripts/09_ablation_gradcam.sh` use `--target_class pred`, `--save_success_only`, `--save_raw_cam`, and `--save_individual_cams`.

The fixed registration template is selected with `src/analysis/select_template.py`. PCA is applied to preprocessed radiographs from the full study dataset. Radiographs are ranked separately by their Euclidean distance from the origin after dividing each retained PCA score by its corresponding singular value and by their PCA reconstruction error. The radiograph with the lowest sum of these two ranks is used as the template. The template selected for the reported analysis belonged to the training partition and is used only as a spatial reference for registration and visualisation.

Grad-CAM is generated from held-out test radiographs and targets the class predicted by each model. A radiograph enters an averaged map only when all five seed-specific models make the same correct prediction. For each seed, the input-space Grad-CAM is normalized to 0–1, the centrally added model-input padding is removed, and the unpadded map is resized to the original radiograph dimensions. The five restored seed maps are then averaged for each radiograph. This per-radiograph mean map is clipped to its 75th–99th percentile range, rescaled to 0–1, and values below 0.15 are set to zero. An ECC affine transformation is estimated between the corresponding radiograph and the fixed template, and the same transformation is applied to its mean Grad-CAM map. Registered maps are then averaged across included radiographs. For the reported class-specific and whole-success average figures, values below 0.15 are suppressed and the heatmap overlay alpha is 0.50.

In the primary DenseNet121 analysis, 315 test radiographs had concordant correct predictions across all five seed-specific models. All 315 maps were successfully registered and included in the averages: 162 younger-group and 153 older-group radiographs.

## Ray masking workflow

The segmentation scripts reproduce the component workflow used for the ray experiments:

1. YOLO detects a component-specific bounding box.
2. A MedSAM model predicts the component mask in the crop.
3. The predicted metacarpal and phalanx masks are visually reviewed and manually corrected where necessary. This quality-control step is performed between scripts 06a and 06b; the reviewed masks are placed under `REFINED_COMPONENT_MASK_ROOT`.
4. `src/segmentation/mask_preprocessing/make_ray_mask.py` merges the reviewed, matched metacarpal and phalanx masks by pixelwise OR.
5. `src/segmentation/mask_preprocessing/apply_mask.py` creates two image sets: **selective** (the specified ray is removed) and **selected** (only the specified ray is retained).

The CNN ablation uses DenseNet121, the same padded preprocessing and five seeds as the baseline. Its checkpoint names are `best_GOLD_ray{2,3,4}_…` for selective inputs and `best_GOLD_ray{2,3,4}_inverted_…` for selected inputs.

## Revision analyses

### Seven-fold cross-validation

The seven-fold workflow uses `StratifiedKFold(shuffle=True, random_state=42)`. Each eligible image appears in exactly one outer test fold; training, validation, and test sets do not overlap within a fold. An inner stratified split uses 17.5% of the outer training partition for validation, giving approximate overall 70%/15%/14.3% train/validation/test proportions. Each fold is trained with the same five seeds and primary hyperparameters. Accuracy, AUC, sensitivity, and specificity are first averaged across the five seed-specific models within each fold. `analysis/kfold/summarize_kfold.py` then reports the mean and sample standard deviation of these fold-level means across the seven folds and writes `outputs/kfold/summary/supplementary_table_s5.csv`.

### Max-AUC versus adapted 1SE checkpoint selection

Script 11 retains the maximum-validation-AUC checkpoint for each seed and also retains the earliest epoch whose validation AUC is at least the maximum validation AUC minus its DeLong standard error. Both selections are evaluated on the unchanged fixed held-out test set using their stored validation-derived thresholds.

### Five-seed mean bootstrap confidence intervals

Script 12 consumes existing primary per-image prediction CSVs; it does not retrain a model. It performs 20,000 nonparametric resamples of held-out test patients separately within the younger and older groups. The same resampled patients are applied to every seed and condition in a replicate. AUC, accuracy, sensitivity, and specificity are calculated separately for each of the five seed models, then averaged. The 2.5th and 97.5th percentiles of these five-seed mean distributions are the 95% confidence limits. No image-level probability or logit ensemble is used.

## Saved checkpoints and evaluation

Training code is in [`src/train/`](./src/train/), fixed-split and fold-level checkpoint inference/evaluation code is in [`src/inference/`](./src/inference/), and Grad-CAM code is in [`src/analysis/`](./src/analysis/). [`weights/CHECKPOINTS_README.md`](./weights/CHECKPOINTS_README.md) lists the expected baseline, ablation CNN, YOLO, and MedSAM checkpoint layout. When model parameters are available, place them in that layout or update the matching variables in `./configs/paths.env`.

```bash
python ./src/inference/inference.py \
  --model_name densenet121 \
  --model_path './weights/baseline/densenet121/best_iter5_densenet121_*_padded_seed*.pth' \
  --data_root ./data/classification \
  --removal ./data/Removal.csv \
  --resize_mode padded --region densenet121_primary \
  --output_csv ./outputs/evaluation/evaluation_results.csv \
  --prediction_csv ./outputs/evaluation/per_image_predictions.csv \
  --ensemble_score_csv ./outputs/evaluation/ensemble_scores.csv \
  --conf_csv ./outputs/evaluation/true_class_confidence.csv \
  --conf_col_name densenet121_primary_true_class_confidence
```

The evaluation output contains seed-level validation/test metrics, per-image predictions, five-seed score tables, and true-class confidence values. It uses the fixed stratified split (seed 42) and the validation-derived threshold stored in each checkpoint.
