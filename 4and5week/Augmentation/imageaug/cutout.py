"""
Cutout augmentation (technique: cutout).

Masks one or more random square regions to zero, forcing a model to rely on
the whole image rather than a few salient pixels.

Usage:
    python cutout.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import random
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from imagelib.cli import parse_args, load_params, augment_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_args("Cutout augmentation")
    params = load_params(args)

    hole_size = int(get_param(params, ["Hole Size", "holeSize", "hole_size", "size"], 16))
    num_holes = int(get_param(params, ["Number of Holes", "numHoles", "num_holes", "holes"], 1))
    copies = int(get_param(params, ["copies", "numCopies"], 1))

    def augment(img: Image.Image) -> Image.Image:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        half = max(1, hole_size // 2)
        for _ in range(num_holes):
            cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            x1, x2 = max(0, cx - half), min(w, cx + half)
            arr[y1:y2, x1:x2] = 0
        return Image.fromarray(arr, mode=img.mode)

    augment_each(
        args.input,
        args.output,
        augment,
        copies=copies,
        extra_summary={"technique": "cutout", "hole_size": hole_size, "num_holes": num_holes},
    )


if __name__ == "__main__":
    main()
