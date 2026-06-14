"""
Color Analysis EDA (analysis: color-analysis).

Aggregates per-channel (R/G/B) intensity histograms across the dataset, the
average colour swatch and the mean image.

Usage:
    python color_analysis.py --input <zip|image> --outdir <dir> --params '{}'
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
from PIL import Image  # noqa: E402

from imagelib.eda import parse_eda_args, load_params, save_fig, save_image, emit  # noqa: E402
from imagelib.io import load_images  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_eda_args("Image colour analysis")
    params = load_params(args)
    sample = int(get_param(params, ["sample", "sampleSize", "max_images"], 300))

    items = load_images(args.input, max_images=sample)
    viz = []

    hist = {c: np.zeros(256, dtype=np.float64) for c in ("R", "G", "B")}
    channel_means = {"R": [], "G": [], "B": []}
    mean_accum = None
    mean_count = 0

    for it in items:
        rgb = it.image.convert("RGB")
        arr = np.asarray(rgb)
        for idx, c in enumerate(("R", "G", "B")):
            channel = arr[..., idx].ravel()
            hist[c] += np.bincount(channel, minlength=256)
            channel_means[c].append(float(channel.mean()))
        # accumulate a 64x64 mean image
        small = np.asarray(rgb.resize((64, 64))).astype(np.float64)
        mean_accum = small if mean_accum is None else mean_accum + small
        mean_count += 1

    # RGB histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, color in zip(("R", "G", "B"), ("#E0524F", "#22A06B", "#4F79E0")):
        ax.plot(hist[c], color=color, label=c)
    ax.set_title("RGB intensity histogram (dataset)")
    ax.set_xlabel("pixel value")
    ax.set_ylabel("frequency")
    ax.legend()
    viz.append(save_fig(fig, args.outdir, "rgb_histogram.png"))

    # Mean image
    if mean_accum is not None and mean_count:
        mean_img = Image.fromarray((mean_accum / mean_count).astype(np.uint8), mode="RGB")
        viz.append(save_image(mean_img.resize((128, 128)), args.outdir, "mean_image.png"))

    avg_color = [round(float(np.mean(channel_means[c])), 1) if channel_means[c] else 0 for c in ("R", "G", "B")]

    emit(viz, {
        "images_analyzed": len(items),
        "average_color_rgb": avg_color,
        "mean_brightness": round(float(np.mean(sum(([*channel_means[c]] for c in ("R", "G", "B")), []))), 1)
        if items else 0,
    })


if __name__ == "__main__":
    main()
