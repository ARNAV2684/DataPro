"""
Shared helpers for the image-pipeline API routes (augment / EDA / model).

These keep each image endpoint thin: parse the JSON summary a script prints,
pull the dataset id out of a storage key, and download the input artifact from
whichever bucket currently holds it.
"""

import json
import time
from typing import Any, Dict, Optional, Tuple


def parse_summary(stdout: str) -> Dict[str, Any]:
    """Return the last JSON object printed on stdout by an image script."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def extract_dataset_id(dataset_key: str) -> str:
    """Pull the dataset UUID out of a storage key, with a safe fallback."""
    try:
        for part in dataset_key.split("/"):
            if part.startswith("dataset_"):
                return part[len("dataset_"):]
        return dataset_key.split("/")[1].replace("dataset_", "")
    except (IndexError, AttributeError):
        return f"images_{int(time.time())}"


def download_image_input(
    supabase_manager,
    storage_key: str,
    buckets: Tuple[str, ...] = ("preprocessed", "augmented", "datasets"),
) -> Tuple[Optional[str], Optional[str]]:
    """Try each bucket in order; return (local_path, bucket_name) for the first hit."""
    for bucket in buckets:
        try:
            result = supabase_manager.download_file(
                bucket_name=supabase_manager.buckets[bucket],
                storage_key=storage_key,
            )
            if result.get("success"):
                return result["local_path"], bucket
        except Exception:
            continue
    return None, None
