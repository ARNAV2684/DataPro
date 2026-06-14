"""
imagelib — shared helpers for the Garuda ML Pipeline image flow.

Image "datasets" are handled as either a single image file or a ZIP archive
(optionally containing class sub-folders, e.g. cats/ and dogs/). These helpers
load such inputs into memory and write processed images back out as a ZIP so
each pipeline stage keeps the existing "one storage key per artifact" model.
"""

from .io import ImageItem, load_images, save_images_zip, IMAGE_EXTENSIONS
from .params import get_param, norm_key

__all__ = [
    "ImageItem",
    "load_images",
    "save_images_zip",
    "IMAGE_EXTENSIONS",
    "get_param",
    "norm_key",
]
