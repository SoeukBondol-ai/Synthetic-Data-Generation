import sys
import urllib.request
import urllib.error
from pathlib import Path
import tarfile
import zipfile
import io

DATASET_DIR = Path(__file__).resolve().parent
FONTS_DIR = DATASET_DIR / "fonts"

def find_fonts_yaml() -> Path:
    for name in ["fonts.yaml", "fonts.yml"]:
        p = DATASET_DIR / name
        if p.exists():
            return p
    return DATASET_DIR / "fonts.yaml"

FONTS_YAML = find_fonts_yaml()

REQUIRED_PACKAGES = ["PIL", "numpy", "tqdm", "yaml"]
MIN_PYTHON = (3, 10)


def check_python() -> bool:
    ok = sys.version_info >= MIN_PYTHON
    status = "OK" if ok else "FAIL"
    print(f"  {status} Python {sys.version.split()[0]}", end="")

    if not ok:
        print(f"  (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+)")
    else:
        print()

    return ok


def check_packages() -> bool:
    all_ok = True

    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            print(f"  OK {pkg}")
        except ImportError:
            print(f"  FAIL {pkg}  - run: pip install {_pip_name(pkg)}")
            all_ok = False

    return all_ok


def _pip_name(import_name: str) -> str:
    return {
        "PIL": "Pillow",
        "yaml": "pyyaml",
    }.get(import_name, import_name)


def load_font_registry() -> list[dict]:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "The 'pyyaml' package is required to parse the font configuration. "
            "Please install it using: pip install pyyaml"
        )

    fonts_yaml_path = find_fonts_yaml()
    if not fonts_yaml_path.exists():
        raise FileNotFoundError(f"Font configuration file (fonts.yaml or fonts.yml) not found at: {fonts_yaml_path}")

    with open(fonts_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "fonts" not in data:
        raise ValueError(f"{fonts_yaml_path.name} must contain a top-level 'fonts:' key")

    return data["fonts"]


_download_cache = {}

def _get_cached_url(url: str) -> bytes:
    if url not in _download_cache:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            _download_cache[url] = response.read()
    return _download_cache[url]


def check_and_download_fonts() -> bool:
    FONTS_DIR.mkdir(exist_ok=True)

    try:
        registry = load_font_registry()
    except Exception as e:
        print(f"  FAIL Could not load fonts.yaml")
        print(f"       {e}")
        return False

    present = [
        entry for entry in registry
        if (FONTS_DIR / entry["file"]).exists()
    ]

    missing = [
        entry for entry in registry
        if not (FONTS_DIR / entry["file"]).exists()
    ]

    print(f"  fonts/ -> {len(present)}/{len(registry)} files present")

    if not missing:
        print("  OK All fonts found - nothing to download")
        return True

    print(f"\n  Downloading {len(missing)} missing font(s)...\n")
    failed = []

    for entry in missing:
        filename = entry.get("file")
        family = entry.get("family", "Unknown family")
        style = entry.get("style", "Unknown style")
        url = entry.get("download_url", "")

        if not filename:
            print("  FAIL Font entry missing 'file' field")
            failed.append("missing file field")
            continue

        dest = FONTS_DIR / filename

        if not url:
            print(f"  FAIL {filename} - no download_url in fonts.yaml")
            failed.append(filename)
            continue

        print(f"  Downloading {filename} ({family} {style})")

        try:
            if url.endswith(".tar.xz") or url.endswith(".tar.gz") or url.endswith(".tar"):
                archive_data = _get_cached_url(url)
                extracted = False
                mode = "r:xz" if url.endswith(".tar.xz") else ("r:gz" if url.endswith(".tar.gz") else "r")
                with tarfile.open(fileobj=io.BytesIO(archive_data), mode=mode) as tf:
                    for member in tf.getmembers():
                        if Path(member.name).name == filename:
                            f = tf.extractfile(member)
                            if f:
                                dest.write_bytes(f.read())
                                extracted = True
                                size_kb = dest.stat().st_size // 1024
                                print(f"      OK saved ({size_kb} KB) [extracted from tarball]")
                                break
                if not extracted:
                    print(f"      FAIL could not find {filename} in tarball archive")
                    failed.append(filename)

            elif url.endswith(".zip"):
                archive_data = _get_cached_url(url)
                extracted = False
                with zipfile.ZipFile(io.BytesIO(archive_data)) as zf:
                    for name in zf.namelist():
                        if Path(name).name == filename:
                            dest.write_bytes(zf.read(name))
                            extracted = True
                            size_kb = dest.stat().st_size // 1024
                            print(f"      OK saved ({size_kb} KB) [extracted from zip]")
                            break
                if not extracted:
                    print(f"      FAIL could not find {filename} in zip archive")
                    failed.append(filename)

            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response:
                    data = response.read()
                dest.write_bytes(data)
                size_kb = dest.stat().st_size // 1024
                print(f"      OK saved ({size_kb} KB)")

        except urllib.error.URLError as e:
            print(f"      FAIL download failed: {e.reason}")
            failed.append(filename)
        except Exception as e:
            print(f"      FAIL error: {e}")
            failed.append(filename)

    if failed:
        print(f"\n  FAIL {len(failed)} font(s) could not be downloaded:")
        for name in failed:
            print(f"       - {name}")

        print("\n  You can install them manually:")
        print("       Linux:  sudo apt install fonts-khmeros fonts-sil-mondulkiri")
        print("       macOS:  brew install --cask font-khmer-os")
        print("       Then copy the .ttf files into the fonts/ folder.")

        return False

    return True



def main() -> None:
    print()
    print("=" * 50)
    print("  Khmer Dataset Generator - Setup")
    print("=" * 50)

    print("\n-- Python version")
    py_ok = check_python()

    print("\n-- Required packages")
    pkg_ok = check_packages()

    print("\n-- Fonts")
    fonts_ok = check_and_download_fonts()

    print("\n" + "=" * 50)

    if py_ok and pkg_ok and fonts_ok:
        print("  OK All good! You can now run:")
        print()
        print("       uv run python generate.py --samples 100 --preview")
        print()
    else:
        print("  FAIL Fix the issues above, then run setup.py again.")

        if not pkg_ok:
            print()
            print("  Quick fix:")
            print("       uv pip install Pillow numpy tqdm pyyaml")

    print("=" * 50)
    print()


if __name__ == "__main__":
    main()