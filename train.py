# Train a CNN to recognize Khmer characters (consonants + digits).
# Runs in Colab/Kaggle where PyTorch is preinstalled, or locally.

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ── Model ─────────────────────────────────────────────────────────────────────

class KhmerCNN(nn.Module):
    """Small CNN for 64×64 grayscale character images (~2.2M params)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64→32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Data ──────────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir: str, img_size: int, batch_size: int, workers: int):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=transform)
    val_ds   = datasets.ImageFolder(Path(data_dir) / "val",   transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=workers)

    # Classes come back sorted by folder name → digits "0".."9" end up first.
    return train_dl, val_dl, train_ds.classes


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen


# ── Main training loop ────────────────────────────────────────────────────────

def train(args) -> tuple[nn.Module, list[str]]:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    train_dl, val_dl, class_names = get_dataloaders(
        args.data, args.img_size, args.batch_size, args.workers
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    model = KhmerCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_dl, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.2%} | "
              f"val loss {va_loss:.4f} acc {va_acc:.2%}")

        if va_acc > best_acc:
            best_acc = va_acc
            _save(model, class_names, args.out_dir)

    print(f"\nBest val accuracy: {best_acc:.2%}")
    _plot(history, args.out_dir)
    return model, class_names


# ── Saving & plotting ─────────────────────────────────────────────────────────

def _save(model: nn.Module, class_names: list[str], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"  saved → {out / 'model.pth'}")


def _plot(history: dict, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # Colab/Kaggle have it; local users may not

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["val_acc"], label="val acc", color="green")
    ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1); ax[1].legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "training.png", dpi=120)
    print(f"  plot saved → {Path(out_dir) / 'training.png'}")


# ── Inference (single image) ──────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, image_path: str, class_names: list[str], device):
    """Predict a single image. Returns (predicted_label, confidence)."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    from PIL import Image
    img = transform(Image.open(image_path).convert("L")).unsqueeze(0).to(device)
    model.eval()
    probs = torch.softmax(model(img), dim=1)[0]
    idx = probs.argmax().item()
    return class_names[idx], probs[idx].item()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Khmer character classifier (Colab/Kaggle friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", "-d", default="./khmer_consonant_dataset",
                   help="Dataset root containing train/ and val/")
    p.add_argument("--out-dir", "-o", default="./model",
                   help="Where to save model.pth and classes.json")
    p.add_argument("--epochs", "-e", type=int, default=10)
    p.add_argument("--batch-size", "-b", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--export-onnx", action="store_true",
                   help="Also export model.onnx (for phone/web deployment)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, class_names = train(args)

    if args.export_onnx:
        device = "cpu"
        model = model.to(device)
        dummy = torch.randn(1, 1, args.img_size, args.img_size, device=device)
        onnx_path = Path(args.out_dir) / "model.onnx"
        torch.onnx.export(model, dummy, onnx_path, input_names=["image"],
                          output_names=["logits"])
        print(f"ONNX saved → {onnx_path}")


if __name__ == "__main__":
    main()
