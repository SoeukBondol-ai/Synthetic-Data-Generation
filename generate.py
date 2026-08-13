# CLI entry point for the Khmer character dataset generator.

import argparse
import sys
import os

# Allow running from this folder directly
sys.path.insert(0, os.path.dirname(__file__))

from generator import generate_dataset, save_preview_grid
from constants import KHMER_CLASSES, KHMER_CONSONANTS, KHMER_DIGITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate a synthetic Khmer handwritten consonant dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default="./khmer_consonant_dataset",
        metavar="DIR",
        help="Root directory to write the dataset into.",
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        metavar="N",
        help="Number of images to generate per consonant class.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        metavar="PX",
        help="Output image resolution in pixels (square).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Fraction of images used for training (rest goes to val).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip train/val split — put everything under 'all/'.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a preview_grid.png showing samples for each class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--classes",
        choices=["all", "consonants", "digits"],
        default="all",
        help="Which character set to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = {
        "all": KHMER_CLASSES,
        "consonants": KHMER_CONSONANTS,
        "digits": KHMER_DIGITS,
    }[args.classes]

    generate_dataset(
        output_dir=args.output,
        samples_per_class=args.samples,
        img_size=args.size,
        train_ratio=0.0 if args.no_split else args.train_ratio,
        seed=args.seed,
        classes=classes,
    )

    if args.preview:
        save_preview_grid(args.output, classes=classes)


if __name__ == "__main__":
    main()