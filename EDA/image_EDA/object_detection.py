"""
Object / Region Detection EDA (analysis: object-detection).

A lightweight, dependency-free region detector: it finds high-edge connected
components and draws bounding boxes around the most prominent regions on a few
sample images. This is meant as a fast, offline EDA aid (no large pretrained
weights to download) -- a deep detector can be swapped in later if desired.

Usage:
    python object_detection.py --input <zip|image> --outdir <dir> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy import ndimage  # noqa: E402

from imagelib.eda import parse_eda_args, load_params, save_image, emit  # noqa: E402
from imagelib.io import load_images  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def _detect_regions(img: Image.Image, max_regions: int):
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    # Edge magnitude via Sobel, then threshold.
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag /= mag.max()
    binary = ndimage.binary_closing(mag > 0.25, iterations=2)
    labels, n = ndimage.label(binary)
    if n == 0:
        return []
    slices = ndimage.find_objects(labels)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    order = np.argsort(sizes)[::-1][:max_regions]
    boxes = []
    h, w = gray.shape
    for idx in order:
        sl = slices[idx]
        if sl is None:
            continue
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop
        # Ignore trivially small regions.
        if (x2 - x1) * (y2 - y1) < 0.01 * h * w:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def main() -> None:
    args = parse_eda_args("Region/object detection (EDA)")
    params = load_params(args)
    sample = int(get_param(params, ["sample", "sampleSize"], 6))
    max_regions = int(get_param(params, ["max_regions", "maxRegions"], 5))

    items = load_images(args.input, max_images=sample)
    viz = []
    total_regions = 0

    for n, it in enumerate(items):
        img = it.image.convert("RGB")
        boxes = _detect_regions(img, max_regions)
        total_regions += len(boxes)
        draw = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        viz.append(save_image(img, args.outdir, f"detected_{n + 1}.png"))

    emit(viz, {
        "images_annotated": len(items),
        "total_regions": total_regions,
        "avg_regions_per_image": round(total_regions / len(items), 2) if items else 0,
        "method": "edge-based connected components",
    })


if __name__ == "__main__":
    main()
