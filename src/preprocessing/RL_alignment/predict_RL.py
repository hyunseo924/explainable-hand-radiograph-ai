import os
import csv
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

INPUT_ROOT = "./data/anon_data_flatten_cropped"
CSV_OUT = "./data/predictions_add.csv"

MODEL_PATH = "./weights/preprocessing/laterality_resnet18_best.pth"
CONF_THRESH = 0.85

IMG_EXTS = (".jpg", ".jpeg", ".png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
])

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

IDX2CLASS = {0: "left", 1: "right"}

rows = []
total_count = 0
low_conf_count = 0

subdirs = [
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d))
]

if subdirs:
    age_groups = subdirs
else:
    age_groups = [None]  

for age in age_groups:

    if age is None:
        age_dir = INPUT_ROOT
        age_name = os.path.basename(INPUT_ROOT)
    else:
        age_dir = os.path.join(INPUT_ROOT, age)
        age_name = age

    for fname in tqdm(sorted(os.listdir(age_dir)),
                      desc=f"Predicting {age_name}"):

        if not fname.lower().endswith(IMG_EXTS):
            continue
        if "bilat" in fname.lower():
            continue

        img_path = os.path.join(age_dir, fname)
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]

        conf, idx = probs.max(dim=0)
        conf = conf.item()
        pred = IDX2CLASS[idx.item()]

        total_count += 1
        if conf < CONF_THRESH:
            low_conf_count += 1

        final = pred if conf >= CONF_THRESH else ""

        rows.append({
            "image_id": fname,
            "age_group": age_name,
            "pred": pred,
            "confidence": round(conf, 4),
            "final": final
        })

rows.sort(key=lambda x: x["confidence"])

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_id", "age_group", "pred", "confidence", "final"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved CSV: {CSV_OUT}")
print(f"Total images processed        : {total_count}")
print(f"Below threshold (< {CONF_THRESH}) : {low_conf_count}")
