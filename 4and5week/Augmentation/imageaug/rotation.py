"""
Random Rotation augmentation (technique: rotation).

For each image, produces rotated copies with a random angle in
[-max_angle, +max_angle], growing the dataset.

Usage:
    python rotation.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import random
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PIL import Image  # noqa: E402

from imagelib.cli import parse_args, load_params, augment_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_args("Random rotation augmentation")
    params = load_params(args)

    max_angle = float(get_param(params, ["Max Angle", "maxAngle", "max_angle", "angle"], 30))
    fill_mode = str(get_param(params, ["Fill Mode", "fillMode", "fill"], "reflect"))
    copies = int(get_param(params, ["copies", "numCopies"], 1))

    def augment(img: Image.Image) -> Image.Image:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)

    augment_each(
        args.input,
        args.output,
        augment,
        copies=copies,
        extra_summary={"technique": "rotation", "max_angle": max_angle, "fill_mode": fill_mode},
    )


if __name__ == "__main__":
    main()
