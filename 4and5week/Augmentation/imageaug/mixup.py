"""
MixUp augmentation (technique: mixup).

Blends pairs of images from the same class to create new convex-combination
samples. ``alpha`` is the blend weight of the partner image (0 = original only,
0.5 = even blend).

Usage:
    python mixup.py --input <zip|image> --output <out.zip> --params '{}'
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "imagelib")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PIL import Image  # noqa: E402

from imagelib.cli import parse_args, load_params  # noqa: E402
from imagelib.io import ImageItem, load_images, save_images_zip, suffix_relpath  # noqa: E402
from imagelib.params import get_param  # noqa: E402


def main() -> None:
    args = parse_args("MixUp augmentation")
    params = load_params(args)
    start = time.time()

    alpha = float(get_param(params, ["Alpha", "alpha"], 0.2))
    blend = min(max(alpha if alpha > 0 else 0.5, 0.0), 1.0)

    items = load_images(args.input)
    out_items = list(items)  # keep originals

    groups = defaultdict(list)
    for item in items:
        groups[item.label].append(item)

    mixed = 0
    skipped_classes = 0
    for label, group in groups.items():
        if len(group) < 2:
            skipped_classes += 1
            continue
        for item in group:
            partner = random.choice([g for g in group if g is not item])
            base = item.image.convert("RGB")
            other = partner.image.convert("RGB").resize(base.size)
            blended = Image.blend(base, other, alpha=blend)
            out_items.append(ImageItem(suffix_relpath(item.relpath, "_mix"), blended, item.label))
            mixed += 1

    save_images_zip(out_items, args.output)

    print(json.dumps({
        "success": True,
        "technique": "mixup",
        "original_size": len(items),
        "augmented_size": len(out_items),
        "mixed_samples": mixed,
        "classes_skipped_too_few": skipped_classes,
        "blend": blend,
        "elapsed_sec": round(time.time() - start, 3),
    }))


if __name__ == "__main__":
    main()
