import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

DATA_ROOT = "./data/pseudo_laterality"
MODEL_OUT = "./weights/preprocessing/laterality_resnet18_best.pth"
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
])

full_ds = ImageFolder(DATA_ROOT, transform=train_tf)
num_total = len(full_ds)
num_val = int(num_total * VAL_RATIO)
num_train = num_total - num_val

train_ds, val_ds = random_split(full_ds, [num_train, num_val])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Class mapping:", full_ds.class_to_idx)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    correct, total = 0, 0
    train_loss = 0.0

    train_pbar = tqdm(
        train_loader,
        desc=f"[Epoch {epoch+1}/{EPOCHS}] Train",
        leave=False
    )

    for x, y in train_pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        train_pbar.set_postfix({
            "loss": f"{train_loss / total:.4f}",
            "acc": f"{correct / total:.4f}"
        })

    train_acc = correct / total
    train_loss /= total

    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    val_pbar = tqdm(
        val_loader,
        desc=f"[Epoch {epoch+1}/{EPOCHS}] Val",
        leave=False
    )

    with torch.no_grad():
        for x, y in val_pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            val_pbar.set_postfix({
                "loss": f"{val_loss / total:.4f}",
                "acc": f"{correct / total:.4f}"
            })

    val_acc = correct / total
    val_loss /= total

    print(f"[{epoch+1}/{EPOCHS}] "
          f"Train Loss {train_loss:.4f} | Acc {train_acc:.4f} || "
          f"Val Loss {val_loss:.4f} | Acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)

print("Training finished. Best Val Acc:", best_val_acc)
