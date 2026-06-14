"""
Format Conversion (preprocessing step: format-conversion).

Converts every image to a consistent colour mode (RGB by default) so downstream
stages and models receive uniform input. The "format"/"quality" parameters are
recorded; the actual on-disk encoding is handled by the ZIP writer.

Usage:
    python format_conversion.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PIL import Image  # noqa: E402

from imagelib.cli import parse_args, load_params, process_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402

_MODE_ALIASES = {
    "rgb": "RGB",
    "grayscale": "L",
    "gray": "L",
    "l": "L",
    "rgba": "RGBA",
}


def main() -> None:
    args = parse_args("Convert images to a consistent colour mode")
    params = load_params(args)

    requested = str(get_param(params, ["format", "mode"], "RGB")).lower()
    target_mode = _MODE_ALIASES.get(requested, "RGB")
    quality = int(get_param(params, ["quality"], 95))

    def transform(img: Image.Image) -> Image.Image:
        if img.mode != target_mode:
            img = img.convert(target_mode)
        return img

    process_each(
        args.input,
        args.output,
        transform,
        extra_summary={
            "operation": "format-conversion",
            "target_mode": target_mode,
            "quality": quality,
        },
    )


if __name__ == "__main__":
    main()
