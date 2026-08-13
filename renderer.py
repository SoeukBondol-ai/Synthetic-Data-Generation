# Renders a single Khmer character as a centered grayscale PIL image.

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import CANVAS_SIZE


def render_character(char: str, font_path: str, font_size: int) -> Image.Image:
    """Draw one character centered on a white square canvas."""
    font   = ImageFont.truetype(font_path, font_size)
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw   = ImageDraw.Draw(canvas)

    # Measure the glyph so we can center it precisely
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (CANVAS_SIZE - text_w) // 2 - bbox[0]
    y = (CANVAS_SIZE - text_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=0)
    return canvas


def is_visible(img: Image.Image, min_dark_pixels: int = 20) -> bool:
    """True if the image has enough dark pixels to be a valid glyph."""
    arr = np.array(img)
    return int((arr < 128).sum()) >= min_dark_pixels