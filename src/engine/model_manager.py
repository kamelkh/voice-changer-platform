"""
AI voice model manager.

Scans the models/ directory for .pth files, provides load/unload
operations, and can download models from a URL.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.utils.constants import MODELS_DIR, RVC_INDEX_EXTENSIONS, RVC_MODEL_EXTENSIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a voice model file."""

    name: str
    path: Path
    index_path: Optional[Path] = None
    description: str = ""
    size_bytes: int = 0
    checksum: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Human-friendly name derived from the file stem."""
        return self.name.replace("_", " ").replace("-", " ").title()

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        return {
            "name": self.name,
            "path": str(self.path),
            "index_path": str(self.index_path) if self.index_path else None,
            "description": self.description,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "tags": self.tags,
        }


class ModelManager:
    """
    Discover, load, and manage RVC voice models.

    Models are stored as .pth files in the ``models/`` directory.
    An optional .index file with the same stem provides FAISS feature
    retrieval for improved voice similarity.
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._catalog: dict[str, ModelInfo] = {}
        self.refresh()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-scan the models directory and rebuild the catalog."""
        self._catalog.clear()
        for ext in RVC_MODEL_EXTENSIONS:
            for pth_path in sorted(self.models_dir.glob(f"*{ext}")):
                name = pth_path.stem
                # Check for a matching index file
                index_path: Optional[Path] = None
                for idx_ext in RVC_INDEX_EXTENSIONS:
                    candidate = pth_path.with_suffix(idx_ext)
                    if candidate.exists():
                        index_path = candidate
                        break

                info = ModelInfo(
                    name=name,
                    path=pth_path,
                    index_path=index_path,
                    size_bytes=pth_path.stat().st_size,
                )
                self._catalog[name] = info

        logger.info("Model catalog refreshed: %d model(s) found.", len(self._catalog))

    def get_models(self) -> list[ModelInfo]:
        """Return all discovered models."""
        return list(self._catalog.values())

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Return model info by name (stem), or *None*."""
        return self._catalog.get(name)

    def get_model_names(self) -> list[str]:
        """Return list of model stem names."""
        return list(self._catalog.keys())

    def has_models(self) -> bool:
        """True when at least one model is available."""
        return bool(self._catalog)

    # ── Download ──────────────────────────────────────────────────────────────

    def download_model(
        self,
        url: str,
        filename: Optional[str] = None,
        expected_checksum: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """
        Download a model from a URL into the models directory.

        Args:
            url:               Direct download URL for the .pth file.
            filename:          Override the download filename.
            expected_checksum: Optional SHA-256 hex digest for verification.

        Returns:
            :class:`ModelInfo` on success, *None* on failure.
        """
        if filename is None:
            filename = url.split("/")[-1].split("?")[0]

        dest_path = self.models_dir / filename
        if dest_path.exists():
            logger.info("Model already exists: %s", dest_path)
            self.refresh()
            return self._catalog.get(dest_path.stem)

        logger.info("Downloading model: %s → %s", url, dest_path)
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            sha256 = hashlib.sha256()

            with open(dest_path, "wb") as fout, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as progress:
                for chunk in response.iter_content(chunk_size=8192):
                    fout.write(chunk)
                    sha256.update(chunk)
                    progress.update(len(chunk))

            actual_checksum = sha256.hexdigest()
            if expected_checksum and actual_checksum != expected_checksum:
                logger.error(
                    "Checksum mismatch for '%s': expected %s, got %s",
                    filename,
                    expected_checksum,
                    actual_checksum,
                )
                dest_path.unlink(missing_ok=True)
                return None

            logger.info("Download complete: %s (SHA-256: %s)", filename, actual_checksum)
            self.refresh()
            return self._catalog.get(dest_path.stem)

        except Exception as exc:
            logger.error("Download failed: %s", exc)
            dest_path.unlink(missing_ok=True)
            return None

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_model(self, name: str) -> bool:
        """
        Delete a model and its index file from disk.

        Args:
            name: Model stem name.

        Returns:
            *True* if the file was deleted.
        """
        info = self._catalog.get(name)
        if info is None:
            logger.warning("Model not found in catalog: %s", name)
            return False

        try:
            info.path.unlink(missing_ok=True)
            if info.index_path:
                info.index_path.unlink(missing_ok=True)
            del self._catalog[name]
            logger.info("Deleted model: %s", name)
            return True
        except Exception as exc:
            logger.error("Failed to delete model '%s': %s", name, exc)
            return False

    # ── Export catalog ────────────────────────────────────────────────────────

    def export_catalog(self, output_path: Optional[Path] = None) -> dict:
        """
        Export the model catalog as a dict (optionally write to JSON).

        Args:
            output_path: When provided, write the JSON to this path.

        Returns:
            Dict representation of the catalog.
        """
        data = {name: info.to_dict() for name, info in self._catalog.items()}
        if output_path is not None:
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
