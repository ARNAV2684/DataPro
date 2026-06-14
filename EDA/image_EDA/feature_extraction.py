"""
Feature Extraction EDA (analysis: feature-extraction).

Builds a simple feature embedding by flattening small thumbnails and projecting
them to 2D with PCA, then plots a scatter coloured by class. This reveals how
separable the classes are in raw-pixel space.

Usage:
    python feature_extraction.py --input <zip|image> --outdir <dir> --params '{}'
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
    args = parse_eda_args("Image feature extraction (PCA embedding)")
    params = load_params(args)
    thumb = int(get_param(params, ["thumb", "thumbnail"], 32))
    sample = int(get_param(params, ["sample", "sampleSize", "max_images"], 500))

    items = load_images(args.input, max_images=sample)
    viz = []

    if len(items) < 2:
        emit(viz, {"images_analyzed": len(items), "note": "Need >= 2 images for a PCA embedding"})
        return

    vectors, labels = [], []
    for it in items:
        arr = np.asarray(it.image.convert("L").resize((thumb, thumb)), dtype=np.float32).ravel() / 255.0
        vectors.append(arr)
        labels.append(it.label if it.label is not None else "unlabeled")
    X = np.vstack(vectors)

    from sklearn.decomposition import PCA  # imported lazily
    n_comp = 2 if X.shape[0] >= 2 and X.shape[1] >= 2 else 1
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    fig, ax = plt.subplots(figsize=(6, 5))
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(unique):
        mask = np.array([l == lab for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.7,
                   color=cmap(i % 10), label=str(lab))
    ax.set_title("PCA embedding of image thumbnails")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(unique) > 1:
        ax.legend(fontsize=8)
    viz.append(save_fig(fig, args.outdir, "pca_embedding.png"))

    emit(viz, {
        "images_analyzed": len(items),
        "thumbnail_size": thumb,
        "num_classes": len(unique),
        "explained_variance_ratio": [round(float(v), 3) for v in pca.explained_variance_ratio_],
    })


if __name__ == "__main__":
    main()
