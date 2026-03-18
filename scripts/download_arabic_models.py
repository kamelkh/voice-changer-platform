#!/usr/bin/env python3
"""
Download Arabic RVC v2 voice models from Hugging Face.

Models included:
  - Fairuz (فيروز)  – Lebanese legendary singer  [MegaChud/Fairuz-200-RVC, 200 epochs]
  - Elissa (إليسا)  – Lebanese-Arab pop singer   [MegaChud/Elissa-150,     150 epochs]
  - Da7ee7 (الدحيح) – Egyptian Arabic YouTuber   [omrahm/voices,           800 epochs]

Usage:
    python scripts/download_arabic_models.py
    python scripts/download_arabic_models.py --models fairuz elissa
    python scripts/download_arabic_models.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Catalogue of Arabic RVC v2 models
# ---------------------------------------------------------------------------

ARABIC_MODELS: list[dict] = [
    {
        "key": "fairuz",
        "name": "Fairuz (فيروز)",
        "description": "Lebanese legendary singer — 200 epochs RVC v2",
        "url": "https://huggingface.co/MegaChud/Fairuz-200-RVC/resolve/main/fairuz.zip",
        "zip_name": "fairuz.zip",
        "expected_pth": "fairuz.pth",   # expected stem after extraction
    },
    {
        "key": "elissa",
        "name": "Elissa (إليسا)",
        "description": "Lebanese-Arab pop singer — 150 epochs RVC v2",
        "url": "https://huggingface.co/MegaChud/Elissa-150/resolve/main/Elissa150.zip",
        "zip_name": "Elissa150.zip",
        "expected_pth": "Elissa150.pth",
    },
    {
        "key": "da7ee7",
        "name": "Da7ee7 (الدحيح)",
        "description": "Egyptian Arabic YouTuber — 800 epochs RVC v2 Mangio-Crepe 40k",
        "url": "https://huggingface.co/omrahm/voices/resolve/main/da7ee7.zip",
        "zip_name": "da7ee7.zip",
        "expected_pth": "da7ee7.pth",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def _download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a progress bar."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Installing requests and tqdm …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm", "-q"])
        import requests
        from tqdm import tqdm

    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=120,
                        headers={"User-Agent": "voice-changer-platform/1.0"})
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    sha = hashlib.sha256()
    with open(dest, "wb") as fout, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=True
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            fout.write(chunk)
            sha.update(chunk)
            bar.update(len(chunk))
    print(f"  Saved → {dest}  (SHA-256: {sha.hexdigest()[:16]}…)")


def _extract_zip(zip_path: Path, models_dir: Path) -> list[Path]:
    """
    Extract all .pth and .index files from *zip_path* flat into *models_dir*.

    Returns the list of extracted paths.
    """
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = Path(member.filename).name          # strip any subdirectory
            if not name or member.is_dir():
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in (".pth", ".index"):
                continue
            dest = models_dir / name
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(dest)
            print(f"  Extracted → {dest}")
    return extracted


def download_model(model: dict, models_dir: Path) -> bool:
    """Download and extract one model.  Returns True on success."""
    models_dir.mkdir(parents=True, exist_ok=True)

    expected_pth = models_dir / model["expected_pth"]
    if expected_pth.exists():
        print(f"[✓] {model['name']} already present: {expected_pth}")
        return True

    zip_path = models_dir / model["zip_name"]
    try:
        # 1. Download zip (skip if already downloaded)
        if not zip_path.exists():
            _download(model["url"], zip_path)
        else:
            print(f"  ZIP already downloaded: {zip_path}")

        # 2. Extract .pth and .index files
        extracted = _extract_zip(zip_path, models_dir)

        # 3. Remove the zip to save disk space
        zip_path.unlink()
        print(f"  Removed zip archive.")

        if not extracted:
            print(f"[✗] No .pth/.index files found inside {zip_path.name}")
            return False

        print(f"[✓] {model['name']} ready.")
        return True

    except Exception as exc:
        print(f"[✗] Failed to download {model['name']}: {exc}")
        zip_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Arabic RVC v2 voice models.")
    parser.add_argument(
        "--models", nargs="+",
        choices=[m["key"] for m in ARABIC_MODELS] + ["all"],
        default=["all"],
        help="Which models to download (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=MODELS_DIR,
        help=f"Target directory for model files (default: {MODELS_DIR})",
    )
    args = parser.parse_args()

    if args.list:
        print("Available Arabic RVC v2 models:")
        for m in ARABIC_MODELS:
            print(f"  {m['key']:10}  {m['name']}  —  {m['description']}")
        return

    keys = (
        [m["key"] for m in ARABIC_MODELS]
        if "all" in args.models
        else args.models
    )
    targets = [m for m in ARABIC_MODELS if m["key"] in keys]

    print(f"Downloading {len(targets)} Arabic voice model(s) to {args.models_dir}\n")
    results = {m["key"]: download_model(m, args.models_dir) for m in targets}

    print("\n── Summary ──────────────────────────────────────")
    for key, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {status}  {key}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
