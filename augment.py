# Augmentation pipeline: rotate → scale → shift → shear → blur → noise → brightness.

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def rotate(img: Image.Image, max_deg: float = 15.0) -> Image.Image:
    """Rotate by a random angle within ±max_deg degrees."""
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)


def scale(img: Image.Image, scale_range: tuple = (0.75, 1.15)) -> Image.Image:
    """Randomly resize and re-center on the original canvas."""
    s = random.uniform(*scale_range)
    w, h = img.size
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    out = Image.new("L", (w, h), 255)
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    out.paste(resized, (x, y))
    return out


def shift(img: Image.Image, max_pixels: int = 8) -> Image.Image:
    """Translate the image by a random offset."""
    dx = random.randint(-max_pixels, max_pixels)
    dy = random.randint(-max_pixels, max_pixels)
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)


def shear(img: Image.Image, max_shear: float = 0.2) -> Image.Image:
    """Apply a random horizontal shear."""
    s = random.uniform(-max_shear, max_shear)
    w, h = img.size
    return img.transform(
        (w, h), Image.AFFINE,
        (1, s, -s * h / 2, 0, 1, 0),
        resample=Image.BICUBIC, fillcolor=255,
    )


def blur(img: Image.Image, max_radius: float = 1.2) -> Image.Image:
    """Apply Gaussian blur with 50% probability."""
    if random.random() < 0.5:
        radius = random.uniform(0.3, max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img


def gaussian_noise(img: Image.Image, std: float = 12.0) -> Image.Image:
    """Add random Gaussian noise to pixel values."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def brightness(img: Image.Image, factor_range: tuple = (0.7, 1.3)) -> Image.Image:
    """Randomly adjust brightness."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_all(img: Image.Image) -> Image.Image:
    """Run the full augmentation pipeline on a single image."""
    img = rotate(img)
    img = scale(img)
    img = shift(img)
    img = shear(img)
    img = blur(img)
    img = gaussian_noise(img)
    img = brightness(img)
    return img