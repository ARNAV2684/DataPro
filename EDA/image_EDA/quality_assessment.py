"""
Quality Assessment EDA (analysis: quality-assessment).

Estimates per-image sharpness (variance of the Laplacian) and brightness, plots
their distributions and flags likely-blurry / over- or under-exposed images.

Usage:
    python quality_assessment.py --input <zip|image> --outdir <dir> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import laplace  # noqa: E402

from imagelib.eda import parse_eda_args, load_params, save_fig, emit  # noqa: E402
from imagelib.io import load_images  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_eda_args("Image quality assessment")
    params = load_params(args)
    blur_threshold = float(get_param(params, ["blur_threshold", "threshold"], 100.0))

    items = load_images(args.input)
    sharpness, brightness = [], []
    for it in items:
        gray = np.asarray(it.image.convert("L"), dtype=np.float64)
        sharpness.append(float(laplace(gray).var()))
        brightness.append(float(gray.mean()))

    viz = []
    if sharpness:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sharpness, bins=20, color="#7B61FF")
        ax.axvline(blur_threshold, color="red", linestyle="--", label=f"blur < {blur_threshold:g}")
        ax.set_title("Sharpness (variance of Laplacian)")
        ax.set_xlabel("sharpness")
        ax.set_ylabel("images")
        ax.legend()
        viz.append(save_fig(fig, args.outdir, "sharpness_distribution.png"))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(brightness, bins=20, color="#FFA94D")
        ax.set_title("Brightness distribution")
        ax.set_xlabel("mean brightness (0-255)")
        ax.set_ylabel("images")
        viz.append(save_fig(fig, args.outdir, "brightness_distribution.png"))

    blurry = sum(1 for s in sharpness if s < blur_threshold)
    dark = sum(1 for b in brightness if b < 50)
    bright = sum(1 for b in brightness if b > 205)

    emit(viz, {
        "images_analyzed": len(items),
        "blur_threshold": blur_threshold,
        "likely_blurry": blurry,
        "too_dark": dark,
        "too_bright": bright,
        "mean_sharpness": round(float(np.mean(sharpness)), 2) if sharpness else 0,
        "mean_brightness": round(float(np.mean(brightness)), 1) if brightness else 0,
    })


if __name__ == "__main__":
    main()
