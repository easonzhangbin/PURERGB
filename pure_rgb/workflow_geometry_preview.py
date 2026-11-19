"""Full-source diagnostic and drawing images in the same source coordinates."""

import numpy as np
from PIL import Image

from .workspace import write_json


def write_geometry_previews(source, directory):
    height, width = source.shape[:2]
    step = max(1, max(height, width) // 1600)
    sample = source[::step, ::step]
    if sample.ndim == 2:
        sample = sample[..., None]
    gray = sample.mean(axis=2)
    low, high = np.quantile(gray, [0.001, 0.999])
    transmission = np.sqrt(np.clip((gray - low) / max(float(high - low), 1e-6), 0, 1))
    diagnostic = directory / "geometry-preview.png"
    _save_preview(transmission, diagnostic)

    # Estimate each channel from inside the film so the rebate does not set
    # the display endpoints. This image is for positioning, not output.
    h, w = sample.shape[:2]
    center = sample[h // 8:h - h // 8, w // 8:w - w // 8]
    low, high = np.quantile(center[::2, ::2], [0.01, 0.99], axis=(0, 1)).astype(np.float32)
    positive = np.sqrt(np.clip((high - sample) / np.maximum(high - low, 1e-6), 0, 1))
    positive_path = directory / "geometry-positive.png"
    _save_preview(positive, positive_path)
    geometry = {
        "preview": str(diagnostic),
        "positive_preview": str(positive_path),
        "width": width,
        "height": height,
        "domain": "Oriented merged transmission; grayscale diagnostic and positive locator",
    }
    write_json(directory / "geometry.json", geometry)
    return geometry


def _save_preview(values, path):
    pixels = np.rint(values * 255).astype(np.uint8)
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[..., 0]
    image = Image.fromarray(pixels)
    image.thumbnail((1600, 1600))
    image.save(path)
