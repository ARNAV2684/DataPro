"""
Lenient parameter lookup.

The frontend sends operation parameters with a variety of key styles depending
on the page (e.g. "targetSize", "Max Angle", "brightness"). These helpers let a
script look up a value by any of several aliases, ignoring case, spaces and
punctuation, so the backend is robust to the exact key the UI happens to send.
"""

from typing import Any, Iterable, Optional


def norm_key(key: str) -> str:
    """Normalise a key to lowercase alphanumerics only ("Max Angle" -> "maxangle")."""
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def get_param(params: Optional[dict], names: Iterable[str], default: Any = None) -> Any:
    """Return the first matching value from ``params`` for any alias in ``names``.

    Args:
        params: The parameter dict (may be None).
        names: One or more candidate keys; matched case/space/punctuation-insensitively.
        default: Returned when no alias matches.
    """
    if not params:
        return default
    normalised = {norm_key(k): v for k, v in params.items()}
    for name in names:
        value = normalised.get(norm_key(name))
        if value is not None:
            return value
    return default
