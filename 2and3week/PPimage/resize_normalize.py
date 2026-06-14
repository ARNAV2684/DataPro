"""
Resize & Normalize (preprocessing step: resize-normalize).

Resizes every image to a standard target size (default 224x224) and optionally
converts to grayscale. The chosen normalization preset (e.g. "imagenet") is
recorded in the summary so the model stage can apply matching mean/std at
training time -- pixel normalization is intentionally NOT baked into the saved
images so they remain viewable previews through the rest of the pipeline.

Usage:
    python resize_normalize.py --input <zip|image> --output <out.zip> --params '{}'
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


def _target_size(params) -> tuple:
    size = get_param(params, ["targetSize", "target_size", "size"], None)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    width = int(get_param(params, ["width", "targetWidth"], 224))
    height = int(get_param(params, ["height", "targetHeight"], 224))
    return width, height


def main() -> None:
    args = parse_args("Resize and standardize images")
    params = load_params(args)

    width, height = _target_size(params)
    grayscale = bool(get_param(params, ["grayscale", "gray"], False))
    normalization = get_param(params, ["normalization", "norm"], "imagenet")

    def transform(img: Image.Image) -> Image.Image:
        if grayscale:
            img = img.convert("L")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img.resize((width, height), Image.BILINEAR)

    process_each(
        args.input,
        args.output,
        transform,
        extra_summary={
            "operation": "resize-normalize",
            "target_size": [width, height],
            "grayscale": grayscale,
            "normalization": normalization,
        },
    )


if __name__ == "__main__":
    main()
