"""
Image classification trainer for the Garuda ML Pipeline.

Trains an image classifier on a ZIP of images organised into class sub-folders
(ImageFolder layout). Supports four architectures selected via --arch:

    cnn-basic           a small CNN trained from scratch
    resnet              torchvision ResNet (18/34/50), optionally pretrained
    efficientnet        torchvision EfficientNet-B0, optionally pretrained
    vision-transformer  torchvision ViT-B/16, optionally pretrained

Writes model.pt, label_map.json, metrics.json and training_curve.png into
--output-dir, and prints a JSON summary plus a "Test Accuracy: X" line.

Usage:
    python image_classifier.py --data-path <zip> --output-dir <dir> \
        --arch cnn-basic --epochs 5 --val-split 0.2 --params '{}'
"""

import argparse
import json
import os
import sys
import time

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

from imagelib.io import load_images  # noqa: E402
from imagelib.params import get_param  # noqa: E402


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    """A small, configurable CNN trained from scratch."""

    def __init__(self, num_classes: int, conv_layers: int = 3,
                 base_filters: int = 32, dropout: float = 0.25):
        super().__init__()
        layers = []
        in_ch = 3
        ch = base_filters
        for _ in range(max(2, conv_layers)):
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = ch
            ch = min(ch * 2, 256)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch: str, num_classes: int, pretrained: bool, hyper: dict):
    """Return (model, input_size, pretrained_used)."""
    arch = (arch or "cnn-basic").lower()

    if arch in ("cnn-basic", "cnn", "basic"):
        model = SimpleCNN(
            num_classes,
            conv_layers=int(get_param(hyper, ["Conv Layers", "conv_layers"], 3)),
            base_filters=int(get_param(hyper, ["Filters", "filters"], 32)),
            dropout=float(get_param(hyper, ["Dropout", "dropout"], 0.25)),
        )
        return model, 64, False

    import torchvision.models as models

    def _weights(name):
        if not pretrained:
            return None
        try:
            return "DEFAULT"
        except Exception:
            return None

    try:
        if "resnet" in arch:
            variant = str(get_param(hyper, ["Architecture", "architecture"], "resnet18")).lower()
            if variant not in ("resnet18", "resnet34", "resnet50"):
                variant = "resnet18"
            model = getattr(models, variant)(weights=_weights(variant))
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model, 224, pretrained
        if "efficientnet" in arch:
            model = models.efficientnet_b0(weights=_weights("efficientnet_b0"))
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model, 224, pretrained
        if "vit" in arch or "transformer" in arch:
            model = models.vit_b_16(weights=_weights("vit_b_16"))
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model, 224, pretrained
    except Exception:
        # Weight download failed (e.g. offline): fall back to randomly-initialised.
        if "resnet" in arch:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model, 224, False
        if "efficientnet" in arch:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model, 224, False
        if "vit" in arch or "transformer" in arch:
            model = models.vit_b_16(weights=None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model, 224, False

    # Unknown arch -> small CNN.
    return SimpleCNN(num_classes), 64, False


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InMemoryImageDataset(Dataset):
    def __init__(self, samples, img_size):
        self.samples = samples  # list of (PIL.Image, label_idx)
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = img.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        return tensor, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Image classifier trainer")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="cnn-basic")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--params", default="{}")
    args, _unknown = parser.parse_known_args()

    try:
        hyper = json.loads(args.params) if args.params else {}
        if not isinstance(hyper, dict):
            hyper = {}
    except (json.JSONDecodeError, TypeError):
        hyper = {}

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    start = time.time()

    # ----- load + label -----
    items = load_images(args.data_path)
    by_label = {}
    for it in items:
        label = it.label if it.label is not None else "unlabeled"
        by_label.setdefault(label, []).append(it.image)

    classes = sorted(k for k, v in by_label.items() if len(v) > 0 and k != "unlabeled") or sorted(by_label.keys())
    if len(classes) < 2:
        sys.stderr.write(
            "Image training needs at least 2 classes (sub-folders). "
            f"Found: {list(by_label.keys())}\n"
        )
        sys.exit(1)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    per_class_counts = {}
    for c in classes:
        imgs = by_label[c][: args.max_per_class]
        per_class_counts[c] = len(imgs)
        for img in imgs:
            samples.append((img, class_to_idx[c]))

    if len(samples) < 4:
        sys.stderr.write("Not enough images to train (need >= 4).\n")
        sys.exit(1)

    # ----- split -----
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * args.val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    pretrained = bool(get_param(hyper, ["Pretrained", "pretrained"], True))
    model, img_size, pretrained_used = build_model(args.arch, len(classes), pretrained, hyper)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        InMemoryImageDataset(train_samples, img_size),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        InMemoryImageDataset(val_samples, img_size),
        batch_size=args.batch_size, shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / max(1, len(train_samples))

        # validation
        model.eval()
        v_running = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_running += criterion(out, yb).item() * xb.size(0)
                all_preds.extend(out.argmax(1).cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
        val_loss = v_running / max(1, len(val_samples))
        val_acc = float(np.mean(np.array(all_preds) == np.array(all_true))) if all_preds else 0.0
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # ----- final metrics -----
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = float(precision_score(all_true, all_preds, average="macro", zero_division=0)) if all_preds else 0.0
    recall = float(recall_score(all_true, all_preds, average="macro", zero_division=0)) if all_preds else 0.0
    f1 = float(f1_score(all_true, all_preds, average="macro", zero_division=0)) if all_preds else 0.0
    final_acc = history["val_acc"][-1] if history["val_acc"] else 0.0

    # ----- save artifacts -----
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({str(i): c for c, i in class_to_idx.items()}, f, indent=2)

    metrics = {
        "test_metrics": {
            "accuracy": round(final_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "arch": args.arch,
        "pretrained_used": pretrained_used,
        "input_size": img_size,
        "num_classes": len(classes),
        "classes": classes,
        "per_class_counts": per_class_counts,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "epochs": args.epochs,
        "history": history,
        "training_time_sec": round(time.time() - start, 2),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # training curve
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs_range, history["train_loss"], label="train loss")
        ax.plot(epochs_range, history["val_loss"], label="val loss")
        ax.plot(epochs_range, history["val_acc"], label="val acc")
        ax.set_xlabel("epoch")
        ax.set_title(f"{args.arch} training")
        ax.legend()
        fig.savefig(os.path.join(args.output_dir, "training_curve.png"), bbox_inches="tight", dpi=110)
        plt.close(fig)
    except Exception:
        pass

    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(json.dumps({"success": True, **metrics["test_metrics"], "arch": args.arch,
                      "num_classes": len(classes), "pretrained_used": pretrained_used}))


if __name__ == "__main__":
    main()
