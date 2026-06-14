"""
Noise Reduction (preprocessing step: noise-reduction).

Removes noise/artifacts using a Gaussian blur or median filter. "strength"
(0..1) is mapped to a filter radius.

Usage:
    python noise_reduction.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PIL import Image, ImageFilter  # noqa: E402

from imagelib.cli import parse_args, load_params, process_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_args("Reduce image noise via blur / median filtering")
    params = load_params(args)

    method = str(get_param(params, ["method"], "gaussian")).lower()
    try:
        strength = float(get_param(params, ["strength", "amount"], 0.5))
    except (TypeError, ValueError):
        strength = 0.5
    strength = max(0.0, min(1.0, strength))

    # Map strength 0..1 to a sensible radius/size.
    radius = round(0.5 + strength * 3.5, 2)          # ~0.5..4.0 px
    median_size = max(3, int(round(strength * 4)) * 2 + 1)  # odd: 3,5,7,9

    def transform(img: Image.Image) -> Image.Image:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if method == "median":
            return img.filter(ImageFilter.MedianFilter(size=median_size))
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    process_each(
        args.input,
        args.output,
        transform,
        extra_summary={
            "operation": "noise-reduction",
            "method": method,
            "strength": strength,
            "radius": radius if method != "median" else median_size,
        },
    )


if __name__ == "__main__":
    main()
