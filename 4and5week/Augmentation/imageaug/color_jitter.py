"""
Color Jittering augmentation (technique: color-jitter).

Randomly perturbs brightness, contrast and saturation within +/- the given
ranges to create lighting-variation copies.

Usage:
    python color_jitter.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import random
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PIL import Image, ImageEnhance  # noqa: E402

from imagelib.cli import parse_args, load_params, augment_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def _rng(params, names, default) -> float:
    try:
        return float(get_param(params, names, default))
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args("Color jitter augmentation")
    params = load_params(args)

    b_range = _rng(params, ["Brightness", "brightness"], 0.2)
    c_range = _rng(params, ["Contrast", "contrast"], 0.2)
    s_range = _rng(params, ["Saturation", "saturation"], 0.2)
    copies = int(get_param(params, ["copies", "numCopies"], 1))

    def augment(img: Image.Image) -> Image.Image:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-b_range, b_range))
        img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-c_range, c_range))
        if img.mode == "RGB":
            img = ImageEnhance.Color(img).enhance(1 + random.uniform(-s_range, s_range))
        return img

    augment_each(
        args.input,
        args.output,
        augment,
        copies=copies,
        extra_summary={
            "technique": "color-jitter",
            "brightness_range": b_range,
            "contrast_range": c_range,
            "saturation_range": s_range,
        },
    )


if __name__ == "__main__":
    main()
