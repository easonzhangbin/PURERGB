"""Select a common post-processing grid before choosing a display sample.

Neighbourhood and position-dependent processors must see the same full output
canvas as a saved TIFF. Point-only controls retain the existing reduced path.
"""

from .positive_post import POST_SPECS
from .professional_controls import checked_controls


def post_requires_full_grid(document):
    if document is None:
        return False
    values = checked_controls(document, POST_SPECS)["values"]
    if any(key.startswith("geometry.") for key in values):
        return True
    if any(
        values.get("lab." + key, 0) > 0
        for key in (
            "sharpen",
            "chroma_denoise",
            "clahe_strength",
            "glow_amount",
            "halation_strength",
        )
    ):
        return True
    return values.get("finish.vignette_stops", 0) != 0
