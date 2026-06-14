"""
Image Statistics EDA (analysis: image-statistics).

Reports dataset composition: images per class, resolution distribution, and
format/mode breakdown, with bar/histogram visualizations.

Usage:
    python image_statistics.py --input <zip|image> --outdir <dir> --params '{}'
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import matplotlib.pyplot as plt  # noqa: E402

from imagelib.eda import parse_eda_args, save_fig, emit  # noqa: E402
from imagelib.io import load_images, class_distribution  # noqa: E402


def main() -> None:
    args = parse_eda_args("Image dataset statistics")
    items = load_images(args.input)

    dist = class_distribution(items)
    formats, modes = {}, {}
    widths, heights = [], []
    for it in items:
        fmt = it.image.format or os.path.splitext(it.relpath)[1].lstrip(".").upper() or "UNKNOWN"
        formats[fmt] = formats.get(fmt, 0) + 1
        modes[it.image.mode] = modes.get(it.image.mode, 0) + 1
        widths.append(it.image.width)
        heights.append(it.image.height)

    viz = []

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(dist.keys()), list(dist.values()), color="#4F79E0")
    ax.set_title("Images per class")
    ax.set_ylabel("count")
    plt.xticks(rotation=30, ha="right")
    viz.append(save_fig(fig, args.outdir, "class_distribution.png"))

    # Resolution scatter
    if widths and heights:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(widths, heights, alpha=0.5, color="#22A06B")
        ax.set_title("Image resolution distribution")
        ax.set_xlabel("width (px)")
        ax.set_ylabel("height (px)")
        viz.append(save_fig(fig, args.outdir, "resolution_scatter.png"))

    def _stats(values):
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 1)}

    emit(viz, {
        "total_images": len(items),
        "num_classes": len(dist),
        "class_distribution": dist,
        "formats": formats,
        "modes": modes,
        "width": _stats(widths),
        "height": _stats(heights),
    })


if __name__ == "__main__":
    main()
