"""
Image Validation (preprocessing step: image-validation).

Scans every image in the dataset, reports format / size / mode distributions
and the per-class image counts, drops unreadable images, and writes the set of
valid images back out as a ZIP so the pipeline can continue.

Usage:
    python image_validation.py --input <zip|image> --output <out.zip> --params '{}'
"""

import json
import os
import sys
import time

# --- locate the workspace root so `imagelib` is importable regardless of cwd ---
_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from imagelib.cli import parse_args  # noqa: E402
from imagelib.io import load_images, save_images_zip, class_distribution  # noqa: E402


def main() -> None:
    args = parse_args("Validate images and report dataset health")
    start = time.time()

    items = load_images(args.input)

    formats: dict = {}
    modes: dict = {}
    widths, heights = [], []
    valid_items = []

    for item in items:
        img = item.image
        try:
            fmt = (img.format or os.path.splitext(item.relpath)[1].lstrip(".").upper() or "UNKNOWN")
            formats[fmt] = formats.get(fmt, 0) + 1
            modes[img.mode] = modes.get(img.mode, 0) + 1
            widths.append(img.width)
            heights.append(img.height)
            valid_items.append(item)
        except Exception:
            continue

    save_images_zip(valid_items, args.output)

    def _stats(values):
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 1),
        }

    summary = {
        "success": True,
        "input_images": len(items),
        "valid_images": len(valid_items),
        "corrupt_images": len(items) - len(valid_items),
        "formats": formats,
        "modes": modes,
        "width": _stats(widths),
        "height": _stats(heights),
        "class_distribution": class_distribution(valid_items),
        "elapsed_sec": round(time.time() - start, 3),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
