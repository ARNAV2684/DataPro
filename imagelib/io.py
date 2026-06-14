"""
I/O helpers for the image pipeline.

Loading rules:
- If the input path is a ZIP archive, every image entry is loaded. The
  immediate parent folder of each image is treated as its class label
  (e.g. "cats/img1.jpg" -> label "cats"). This matches the standard
  ImageFolder layout used for image classification.
- Otherwise the input is treated as a single image file (no label).

Saving always produces a ZIP, preserving each image's relative path so the
class-folder structure flows through every stage of the pipeline.
"""

import io
import os
import zipfile
from pathlib import Path
from typing import List, Optional

from PIL import Image

# Extensions we treat as images inside an archive.
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif",
}


class ImageItem:
    """A single image plus its archive-relative path and (optional) class label."""

    def __init__(self, relpath: str, image: Image.Image, label: Optional[str] = None):
        self.relpath = relpath
        self.image = image
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ImageItem(relpath={self.relpath!r}, label={self.label!r}, size={self.image.size})"


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _label_from_relpath(relpath: str) -> Optional[str]:
    """Use the immediate parent directory as the class label, if any."""
    parts = Path(relpath).parts
    if len(parts) >= 2:
        return parts[-2]
    return None


def load_images(input_path, max_images: Optional[int] = None) -> List[ImageItem]:
    """Load images from a ZIP archive or a single image file.

    Args:
        input_path: Path to a .zip archive or a single image file.
        max_images: Optional cap on the number of images loaded (useful for
            previews / quick analyses on large datasets).

    Returns:
        A list of ImageItem. Corrupt entries inside an archive are skipped.
    """
    input_path = str(input_path)
    items: List[ImageItem] = []

    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as zf:
            names = [
                n for n in zf.namelist()
                if not n.endswith("/") and _is_image_name(n)
                and not os.path.basename(n).startswith(".")
            ]
            names.sort()
            for name in names:
                if max_images is not None and len(items) >= max_images:
                    break
                try:
                    with zf.open(name) as f:
                        img = Image.open(io.BytesIO(f.read()))
                        img.load()
                except Exception:
                    # Skip unreadable / corrupt images rather than failing the run.
                    continue
                items.append(ImageItem(name, img, _label_from_relpath(name)))
    else:
        img = Image.open(input_path)
        img.load()
        items.append(ImageItem(Path(input_path).name, img, None))

    return items


def _format_for(relpath: str) -> str:
    ext = Path(relpath).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".bmp":
        return "BMP"
    if ext in (".tiff", ".tif"):
        return "TIFF"
    if ext == ".webp":
        return "WEBP"
    return "PNG"


def save_images_zip(items: List[ImageItem], output_path) -> str:
    """Write a list of ImageItem to a ZIP archive, preserving relative paths."""
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for it in items:
            fmt = _format_for(it.relpath)
            img = it.image
            # JPEG/BMP cannot hold alpha; convert defensively.
            if fmt in ("JPEG", "BMP") and img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            save_kwargs = {}
            if fmt == "JPEG":
                save_kwargs["quality"] = 95
            img.save(buf, format=fmt, **save_kwargs)
            zf.writestr(it.relpath, buf.getvalue())

    return output_path


def class_distribution(items: List[ImageItem]) -> dict:
    """Return {label: count} for the given items (None labels grouped as 'unlabeled')."""
    dist: dict = {}
    for it in items:
        key = it.label if it.label is not None else "unlabeled"
        dist[key] = dist.get(key, 0) + 1
    return dist
