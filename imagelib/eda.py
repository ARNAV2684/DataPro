"""
Common scaffolding for image EDA scripts.

Unlike preprocess/augment (which emit a single ZIP), EDA scripts emit one or
more visualization PNGs into an output directory plus a JSON ``stats`` blob.
The API route uploads every PNG to the ``eda`` bucket and returns their public
URLs for the frontend to render.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt  # noqa: E402

from .io import load_images  # noqa: E402,F401  (re-exported for scripts)
from .params import get_param  # noqa: E402,F401


def parse_eda_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Path to input zip or image")
    parser.add_argument("--outdir", required=True, help="Directory to write visualizations into")
    parser.add_argument("--params", default="{}", help="JSON string of analysis parameters")
    args, _unknown = parser.parse_known_args()
    return args


def load_params(args: argparse.Namespace) -> dict:
    try:
        params = json.loads(args.params) if args.params else {}
        return params if isinstance(params, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def save_fig(fig, outdir: str, name: str) -> str:
    """Save a matplotlib figure into outdir and return its filename."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), bbox_inches="tight", dpi=110)
    plt.close(fig)
    return name


def save_image(img, outdir: str, name: str) -> str:
    """Save a PIL image into outdir and return its filename."""
    os.makedirs(outdir, exist_ok=True)
    img.save(os.path.join(outdir, name))
    return name


def emit(visualizations, stats) -> None:
    """Print the JSON result consumed by the API route."""
    print(json.dumps({
        "success": True,
        "visualizations": visualizations,
        "stats": stats,
    }))
