"""
Color Correction (preprocessing step: color-correction).

Adjusts brightness, contrast and saturation using PIL's ImageEnhance. Each
value is a multiplier where 1.0 leaves the image unchanged.

Usage:
    python color_correction.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PIL import Image, ImageEnhance  # noqa: E402

from imagelib.cli import parse_args, load_params, process_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def _factor(params, names, default=1.0) -> float:
    try:
        return float(get_param(params, names, default))
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args("Adjust image brightness, contrast and saturation")
    params = load_params(args)

    brightness = _factor(params, ["brightness"], 1.0)
    contrast = _factor(params, ["contrast"], 1.0)
    saturation = _factor(params, ["saturation", "color"], 1.0)

    def transform(img: Image.Image) -> Image.Image:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        # Saturation only applies to colour images.
        if saturation != 1.0 and img.mode == "RGB":
            img = ImageEnhance.Color(img).enhance(saturation)
        return img

    process_each(
        args.input,
        args.output,
        transform,
        extra_summary={
            "operation": "color-correction",
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
        },
    )


if __name__ == "__main__":
    main()
