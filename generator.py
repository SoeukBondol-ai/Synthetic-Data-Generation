import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from augment import apply_all
from constants import (
    FONT_PATHS,
    FONT_SIZE_RANGE,
    IMG_SIZE,
    KHMER_CONSONANTS,
    LABEL_MAP,
)
from renderer import is_visible, render_character


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str,
    samples_per_class: int = 100,
    img_size: int = IMG_SIZE,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic Khmer consonant image dataset.

    Folder structure produced:
        <output_dir>/
            train/<class_name>/<class_name>_NNNN.png
            val/<class_name>/<class_name>_NNNN.png
            dataset_info.json

    Args:
        output_dir:        Root directory to write the dataset into.
        samples_per_class: Number of images to generate per consonant class.
        img_size:          Final image resolution in pixels (square).
        train_ratio:       Fraction of images used for training (rest → val).
        seed:              Random seed for reproducibility.

    Returns:
        Metadata dict also written to dataset_info.json.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    _print_header(samples_per_class, img_size, output_dir, train_ratio)

    # Create all class folders up front
    for split in ("train", "val"):
        for _, name in KHMER_CONSONANTS:
            (output_dir / split / name).mkdir(parents=True, exist_ok=True)

    stats: dict = {"total": 0, "per_class": {}}
    total_expected = len(KHMER_CONSONANTS) * samples_per_class

    with tqdm(total=total_expected, desc="Generating", unit="img") as pbar:
        for char, name in KHMER_CONSONANTS:
            images = _generate_class(char, samples_per_class, img_size, pbar)
            _save_split(images, output_dir, name, train_ratio)
            n_train = int(len(images) * train_ratio)
            stats["per_class"][name] = {
                "char":  char,
                "count": len(images),
                "train": n_train,
                "val":   len(images) - n_train,
            }
            stats["total"] += len(images)

    meta = _save_metadata(output_dir, img_size, stats)
    _print_footer(stats["total"], output_dir)
    return meta


def _generate_class(
    char: str,
    samples: int,
    img_size: int,
    pbar: tqdm,
) -> list[Image.Image]:
    """Render + augment `samples` images for a single character."""
    images: list[Image.Image] = []
    max_attempts = samples * 10
    attempts = 0

    while len(images) < samples and attempts < max_attempts:
        attempts += 1
        font_path = random.choice(FONT_PATHS)
        font_size = random.randint(*FONT_SIZE_RANGE)
        try:
            img = render_character(char, font_path, font_size)
            if not is_visible(img):
                continue
            img = apply_all(img)
            img = img.resize((img_size, img_size), Image.LANCZOS)
            images.append(img)
            pbar.update(1)
        except Exception:
            continue

    return images


def _save_split(
    images: list[Image.Image],
    output_dir: Path,
    name: str,
    train_ratio: float,
) -> None:
    """Shuffle and save images into train / val folders."""
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    split_map = {"train": images[:n_train], "val": images[n_train:]}
    for split, imgs in split_map.items():
        for i, img in enumerate(imgs):
            path = output_dir / split / name / f"{name}_{i:04d}.png"
            img.save(path)


def _save_metadata(output_dir: Path, img_size: int, stats: dict) -> dict:
    """Write dataset_info.json and return the metadata dict."""
    meta = {
        "description": "Synthetic Khmer consonant dataset",
        "num_classes":  len(KHMER_CONSONANTS),
        "image_size":   img_size,
        "fonts_used":   [os.path.basename(f) for f in FONT_PATHS],
        "total_images": stats["total"],
        "label_map":    LABEL_MAP,
        "classes":      stats["per_class"],
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


# ── Preview grid ───────────────────────────────────────────────────────────────

def save_preview_grid(
    output_dir: str,
    n_samples: int = 8,
    save_path: str | None = None,
) -> str:
    """
    Save a PNG grid showing all 33 consonants × n_samples example images.

    Args:
        output_dir: Root of the generated dataset (must contain train/).
        n_samples:  Number of sample images to show per row.
        save_path:  Override the output file path (optional).

    Returns:
        Path to the saved grid image.
    """
    output_dir = Path(output_dir)
    cell       = IMG_SIZE + 4
    label_w    = 52

    grid_w = label_w + n_samples * cell + 20
    grid_h = len(KHMER_CONSONANTS) * cell + 20
    grid   = Image.new("RGB", (grid_w, grid_h), (245, 245, 250))
    draw   = ImageDraw.Draw(grid)

    try:
        label_font = ImageFont.truetype(FONT_PATHS[0], 30)
    except Exception:
        label_font = ImageFont.load_default()

    for row, (char, name) in enumerate(KHMER_CONSONANTS):
        y = row * cell + 10
        if row % 2 == 0:
            draw.rectangle([0, y - 2, grid_w, y + cell - 2], fill=(235, 240, 248))
        draw.text((6, y + 10), char, font=label_font, fill=(30, 30, 30))

        folder = output_dir / "train" / name
        sample_paths = sorted(folder.glob("*.png"))[:n_samples]
        for col, img_path in enumerate(sample_paths):
            x = label_w + col * cell + 4
            img = Image.open(img_path).convert("RGB")
            grid.paste(img, (x, y + 2))

    save_path = save_path or str(output_dir / "preview_grid.png")
    grid.save(save_path)
    print(f" Preview saved → {save_path}")
    return save_path


# ── Internal helpers ───────────────────────────────────────────────────────────

def _print_header(samples: int, img_size: int, output_dir: Path, train_ratio: float) -> None:
    val_pct   = int((1 - train_ratio) * 100)
    train_pct = int(train_ratio * 100)
    print(f"\n{'='*52}")
    print(f"  Khmer Consonant Dataset Generator")
    print(f"{'='*52}")
    print(f"  Classes     : {len(KHMER_CONSONANTS)} consonants")
    print(f"  Fonts       : {len(FONT_PATHS)} fonts")
    print(f"  Samples/cls : {samples}  →  {len(KHMER_CONSONANTS) * samples} total")
    print(f"  Image size  : {img_size}×{img_size} px")
    print(f"  Split       : train {train_pct}% / val {val_pct}%")
    print(f"  Output      : {output_dir}")
    print(f"{'='*52}\n")


def _print_footer(total: int, output_dir: Path) -> None:
    print(f"\n{'='*52}")
    print(f"{total} images generated")
    print(f"{output_dir}/train  &  {output_dir}/val")
    print(f"{output_dir}/dataset_info.json")
    print(f"{'='*52}\n")