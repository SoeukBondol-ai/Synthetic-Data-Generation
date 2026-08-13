from pathlib import Path
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent          # dataset/
FONTS_DIR   = DATASET_DIR / "fonts"          # dataset/fonts/

def _find_fonts_yaml() -> Path:
    for name in ["fonts.yaml", "fonts.yml"]:
        p = DATASET_DIR / name
        if p.exists():
            return p
    return DATASET_DIR / "fonts.yaml"

FONTS_YAML = _find_fonts_yaml()

# ── Load font list from config ─────────────────────────────────────────────────
def _load_font_paths() -> list[str]:
    """Read fonts.yaml → absolute paths to fonts present in fonts/."""
    with open(FONTS_YAML, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = []
    missing = []
    for entry in config["fonts"]:
        path = FONTS_DIR / entry["file"]
        if path.exists():
            paths.append(str(path))
        else:
            missing.append(entry["file"])

    if missing:
        print(f"[fonts] Warning: {len(missing)} font(s) not found in fonts/ → {missing}")
    if not paths:
        raise FileNotFoundError(
            f"No fonts found in {FONTS_DIR}.\n"
            "Make sure the fonts/ folder is present (it should be committed to the repo)."
        )
    return paths

FONT_PATHS: list[str] = _load_font_paths()

# ── 33 Khmer consonants in Unicode order ──────────────────────────────────────
# Each entry: (unicode_char, folder_name)
# folder_name is used as the class label when training.
KHMER_CONSONANTS: list[tuple[str, str]] = [
    ("ក", "ka"),   ("ខ", "kha"),  ("គ", "ko"),   ("ឃ", "kho"),
    ("ង", "ngo"),  ("ច", "cha"),  ("ឆ", "chha"), ("ជ", "cho"),
    ("ឈ", "chho"), ("ញ", "nyo"),  ("ដ", "da"),   ("ឋ", "tha"),
    ("ឌ", "do"),   ("ឍ", "dho"),  ("ណ", "na"),   ("ត", "ta"),
    ("ថ", "tha2"), ("ទ", "to"),   ("ធ", "tho"),  ("ន", "no"),
    ("ប", "ba"),   ("ផ", "pha"),  ("ព", "po"),   ("ភ", "pho"),
    ("ម", "mo"),   ("យ", "yo"),   ("រ", "ro"),   ("ល", "lo"),
    ("វ", "vo"),   ("ស", "so"),   ("ហ", "ho"),   ("ឡ", "la"),
    ("អ", "a"),
]

# ── 10 Khmer digits in Unicode order ─────────────────────────────────────────
# Each entry: (unicode_char, folder_name)
# folder_name is the numeric value, so labels map naturally to int(digit).
KHMER_DIGITS: list[tuple[str, str]] = [
    ("០", "0"), ("១", "1"), ("២", "2"), ("៣", "3"), ("៤", "4"),
    ("៥", "5"), ("៦", "6"), ("៧", "7"), ("៨", "8"), ("៩", "9"),
]

# Combined character set (consonants + digits) — used by default.
KHMER_CLASSES: list[tuple[str, str]] = KHMER_CONSONANTS + KHMER_DIGITS

# Convenient lookup: folder_name → integer index (for model label encoding)
LABEL_MAP: dict[str, int] = {
    name: idx for idx, (_, name) in enumerate(KHMER_CLASSES)
}

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE        = 64        # Final output resolution (px) — matches existing project
CANVAS_SIZE     = 96        # Larger render canvas to avoid clipping during rotation
FONT_SIZE_RANGE = (48, 72)  # Random font size range per sample