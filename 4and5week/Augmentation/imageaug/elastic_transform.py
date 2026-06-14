"""
Elastic Transform augmentation (technique: elastic-transform).

Applies smooth random displacement fields to deform images elastically
(classic for handwriting / medical image augmentation).

- alpha controls displacement magnitude.
- sigma controls how smooth (correlated) the displacement is.

Usage:
    python elastic_transform.py --input <zip|image> --output <out.zip> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: E402

from imagelib.cli import parse_args, load_params, augment_each  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_args("Elastic transform augmentation")
    params = load_params(args)

    alpha = float(get_param(params, ["Alpha", "alpha"], 1.0))
    sigma = float(get_param(params, ["Sigma", "sigma"], 0.5))
    copies = int(get_param(params, ["copies", "numCopies"], 1))

    # Map the UI ranges (alpha 0..5, sigma 0.1..2) to stable pixel magnitudes.
    magnitude = alpha * 4.0          # ~px of displacement
    smoothing = max(0.1, sigma) * 8.0  # smoothing of the field

    def augment(img: Image.Image) -> Image.Image:
        is_gray = img.mode == "L"
        if not is_gray and img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img)
        shape = arr.shape[:2]

        dx = gaussian_filter(np.random.rand(*shape) * 2 - 1, smoothing) * magnitude
        dy = gaussian_filter(np.random.rand(*shape) * 2 - 1, smoothing) * magnitude
        yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = (np.reshape(yy + dy, (-1,)), np.reshape(xx + dx, (-1,)))

        if arr.ndim == 3:
            out = np.zeros_like(arr)
            for ch in range(arr.shape[2]):
                out[..., ch] = map_coordinates(
                    arr[..., ch], indices, order=1, mode="reflect"
                ).reshape(shape)
            return Image.fromarray(out.astype(np.uint8), mode="RGB")
        out = map_coordinates(arr, indices, order=1, mode="reflect").reshape(shape)
        return Image.fromarray(out.astype(np.uint8), mode="L")

    augment_each(
        args.input,
        args.output,
        augment,
        copies=copies,
        extra_summary={"technique": "elastic-transform", "alpha": alpha, "sigma": sigma},
    )


if __name__ == "__main__":
    main()
