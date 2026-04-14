"""Runtime compatibility shims for legacy dependencies.

This project still relies on detectron2==0.6, which imports Pillow resampling
constants removed in Pillow 10. Importing ``sitecustomize`` happens
automatically during Python startup, so patch the aliases here before any
third-party imports run.
"""

from PIL import Image


def _ensure_pillow_resampling_aliases() -> None:
    aliases = {
        "LINEAR": "BILINEAR",
        "CUBIC": "BICUBIC",
        "ANTIALIAS": "LANCZOS",
    }
    for legacy_name, current_name in aliases.items():
        if not hasattr(Image, legacy_name) and hasattr(Image, current_name):
            setattr(Image, legacy_name, getattr(Image, current_name))


_ensure_pillow_resampling_aliases()
