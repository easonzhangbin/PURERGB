"""Suggest a shared film-base rectangle from the three RAW locator previews."""

from itertools import product

import numpy as np


def _runs(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(changes[::2], changes[1::2])


def _bands(image):
    """Find thin bright bands between two sustained picture edges."""
    height, width = image.shape
    top, bottom = round(0.22 * height), round(0.78 * height)
    strip = image[top:bottom].astype(np.float32)
    profile = np.median(strip, axis=0)
    edge = max(1, round(width * 0.003))
    steps = np.pad(profile[2 * edge:] - profile[:-2 * edge], edge)
    margin = max(12, round(0.055 * width))
    steps[:margin] = steps[-margin:] = 0
    minimum, maximum = max(5, round(0.007 * width)), round(0.08 * width)
    shoulder = max(8, round(0.028 * width))
    threshold = max(3.0, 0.07 * np.ptp(np.percentile(profile, (10, 90))))
    rises = [start + np.argmax(steps[start:end]) for start, end in _runs(steps >= 2)]
    falls = [start + np.argmin(steps[start:end]) for start, end in _runs(steps <= -2)]
    rises = sorted(rises, key=lambda x: steps[x], reverse=True)[:24]
    falls = sorted(falls, key=lambda x: steps[x])[:24]
    matches = []
    for left, right in product(rises, falls):
        span = right - left
        if not minimum <= span <= maximum:
            continue
        if max(steps[left], -steps[right]) < 2 * threshold:
            continue
        # Leave room for a transition pixel, scanner blur and a mild tilt.
        inset = max(1, round(span * 0.22))
        x0, x1 = left + inset, right - inset + 1
        a, b = left - 2, right + 3
        before = strip[:, a - shoulder:a]
        after = strip[:, b:b + shoulder]
        if np.median(profile[x0:x1]) - max(np.median(profile[a - shoulder:a]),
                                          np.median(profile[b:b + shoulder])) < threshold:
            continue
        # A flat bright margin alongside one picture lacks texture on its other side.
        if min(np.ptp(np.percentile(side, (10, 90))) for side in (before, after)) < max(2, threshold * 0.5):
            continue
        core = strip[:, x0:x1]
        separation = np.median(core, axis=1) - np.maximum(np.median(before, axis=1), np.median(after, axis=1))
        coherent = separation >= threshold
        if np.mean(coherent) < 0.73:
            continue
        # Exclude rows crossed by marks, edges or the start of a perforation.
        clean = coherent & (np.ptp(core, axis=1) <= np.maximum(5, separation * 0.35))
        matches.append((x0, x1, clean, float(np.median(separation))))
    return top, sorted(matches, key=lambda item: item[-1], reverse=True)[:12]


def suggest_frame_gap(previews):
    """Return a conservative normalized ROI, or None if three channels do not agree.

    The previews are unrotated, same-grid locator images. The selection is only
    a visual suggestion. The CFA meter and operator confirmation are separate.
    """
    images = tuple(previews)
    if len(images) != 3 or any(image is None or image.ndim != 2 for image in images):
        return None
    if min(images[0].shape) < 100 or any(image.shape != images[0].shape for image in images):
        return None

    proposals = []
    for transpose in (False, True):
        oriented = tuple(image.T if transpose else image for image in images)
        height, width = oriented[0].shape
        top, red = _bands(oriented[0])
        _, green = _bands(oriented[1])
        _, blue = _bands(oriented[2])
        for channels in product(red, green, blue):
            x0 = max(band[0] for band in channels)
            x1 = min(band[1] for band in channels)
            if x1 - x0 < 3:
                continue
            common = np.logical_and.reduce([band[2] for band in channels])
            runs = list(_runs(common))
            if not runs:
                continue
            y0, y1 = max(runs, key=lambda run: run[1] - run[0])
            if y1 - y0 < 0.19 * height:
                continue
            # Keep the sample away from the ends of the supported run as well.
            padding = max(2, round((y1 - y0) * 0.15))
            x0, x1 = x0 / width, x1 / width
            y0, y1 = (top + y0 + padding) / height, (top + y1 - padding) / height
            area = (y0, x0, y1, x1) if transpose else (x0, y0, x1, y1)
            score = min(band[-1] for band in channels) * (y1 - y0)
            proposals.append((score, area))

    if not proposals:
        return None
    proposals.sort(reverse=True)
    score, best = proposals[0]
    for runner_up, area in proposals[1:]:
        if runner_up < score * 0.8:
            break
        if min(area[2], best[2]) <= max(area[0], best[0]) or min(area[3], best[3]) <= max(area[1], best[1]):
            return None
    return best
