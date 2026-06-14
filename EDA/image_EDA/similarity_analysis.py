"""
Similarity Analysis EDA (analysis: similarity-analysis).

Computes pairwise cosine similarity between a sample of image thumbnails and
renders a similarity heatmap, reporting the most-similar pair.

Usage:
    python similarity_analysis.py --input <zip|image> --outdir <dir> --params '{}'
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

from imagelib.eda import parse_eda_args, load_params, save_fig, emit  # noqa: E402
from imagelib.io import load_images  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_eda_args("Image similarity analysis")
    params = load_params(args)
    sample = int(get_param(params, ["sample", "sampleSize", "max_images"], 24))
    thumb = int(get_param(params, ["thumb", "thumbnail"], 32))

    items = load_images(args.input, max_images=sample)
    viz = []

    if len(items) < 2:
        emit(viz, {"images_analyzed": len(items), "note": "Need >= 2 images for similarity"})
        return

    vectors = []
    for it in items:
        arr = np.asarray(it.image.convert("L").resize((thumb, thumb)), dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        vectors.append(arr / norm if norm > 0 else arr)
    X = np.vstack(vectors)
    sim = X @ X.T  # cosine similarity (vectors are L2-normalised)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Cosine similarity ({len(items)} images)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    viz.append(save_fig(fig, args.outdir, "similarity_heatmap.png"))

    # Most similar off-diagonal pair
    masked = sim.copy()
    np.fill_diagonal(masked, -1)
    i, j = np.unravel_index(np.argmax(masked), masked.shape)

    emit(viz, {
        "images_analyzed": len(items),
        "most_similar_pair": [items[i].relpath, items[j].relpath],
        "max_similarity": round(float(masked[i, j]), 4),
        "mean_similarity": round(float(sim[np.triu_indices(len(items), k=1)].mean()), 4),
    })


if __name__ == "__main__":
    main()
