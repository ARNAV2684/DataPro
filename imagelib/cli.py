"""
Common CLI scaffolding for image pipeline scripts.

Every preprocessing / augmentation script accepts the same three arguments
(``--input``, ``--output``, ``--params``) and applies a per-image transform.
``--params`` is a JSON object forwarded verbatim from the API request, parsed
leniently via :func:`imagelib.params.get_param`.
"""

import argparse
import json
import time
from typing import Callable, List, Optional

from PIL import Image

from .io import ImageItem, load_images, save_images_zip


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Path to input zip or image file")
    parser.add_argument("--output", required=True, help="Path to output zip file")
    parser.add_argument("--params", default="{}", help="JSON string of operation parameters")
    # Tolerate unknown flags so callers can pass extras without breaking the script.
    args, _unknown = parser.parse_known_args()
    return args


def load_params(args: argparse.Namespace) -> dict:
    try:
        params = json.loads(args.params) if args.params else {}
        return params if isinstance(params, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def process_each(
    input_path: str,
    output_path: str,
    transform: Callable[[Image.Image], Image.Image],
    extra_summary: Optional[dict] = None,
    max_images: Optional[int] = None,
) -> dict:
    """Apply ``transform`` to every image and write the results to a ZIP.

    Prints a JSON summary to stdout (consumed by the API route) and returns it.
    """
    start = time.time()
    items = load_images(input_path, max_images=max_images)
    out_items: List[ImageItem] = []
    errors = 0

    for item in items:
        try:
            item.image = transform(item.image)
            out_items.append(item)
        except Exception:
            errors += 1

    save_images_zip(out_items, output_path)

    summary = {
        "success": True,
        "input_images": len(items),
        "output_images": len(out_items),
        "errors": errors,
        "elapsed_sec": round(time.time() - start, 3),
    }
    if extra_summary:
        summary.update(extra_summary)

    print(json.dumps(summary))
    return summary
