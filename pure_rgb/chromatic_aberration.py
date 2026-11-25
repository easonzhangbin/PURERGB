"""Conservative, colour-preserving lateral CA in the green channel's geometry.

Estimate only a small radial magnification, from independent textured tiles.
Natural colour differences are never removed by changing saturation or gains.
"""

import math

import cv2
import numpy as np


POLICY = "green-relative-lateral-ca-v1"


def _feature(plane):
    value = np.log(np.maximum(plane, 1e-5)).astype(np.float32)
    value = cv2.GaussianBlur(value, (0, 0), 0.8)
    gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _fit(reference, carrier, center):
    h, w = reference.shape
    side = min(256, max(64, min(h, w) // 5))
    half = side // 2
    window = cv2.createHanningWindow((side, side), cv2.CV_32F)
    samples = []
    for cy in np.linspace(half + 6, h - half - 7, 5):
        for cx in np.linspace(half + 6, w - half - 7, 5):
            x, y = int(cx) - half, int(cy) - half
            a = reference[y:y + side, x:x + side].copy()
            b = carrier[y:y + side, x:x + side].copy()
            if min(float(a.std()), float(b.std())) < 0.01:
                continue
            a -= a.mean()
            b -= b.mean()
            (dx, dy), response = cv2.phaseCorrelate(a, b, window)
            if (not all(math.isfinite(v) for v in (dx, dy, response))
                    or response < 0.15 or max(abs(dx), abs(dy)) > 8):
                continue
            samples.append((cx - center[0], cy - center[1], dx, dy, response))
    if len(samples) < 10:
        return dict(status="insufficient", tiles=len(samples), scale=1.0)
    values = np.asarray(samples, dtype=np.float64)
    locations, shifts = values[:, :2], values[:, 2:4]
    weights = np.minimum(values[:, 4], 1.0)
    # A shared radial scale must explain both axes. A residual translation is
    # fitted for diagnostics only; it is never silently substituted for CA.
    use = np.ones(len(values), dtype=bool)
    coefficient = 0.0
    for _ in range(4):
        p, d, weight = locations[use], shifts[use], weights[use]
        if len(p) < 10:
            return dict(status="inconsistent", tiles=int(use.sum()), scale=1.0)
        pc = np.average(p, axis=0, weights=weight)
        dc = np.average(d, axis=0, weights=weight)
        coefficient = float(np.sum(weight[:, None] * (p - pc) * (d - dc))
                            / max(np.sum(weight[:, None] * (p - pc) ** 2), 1e-12))
        translation = dc - coefficient * pc
        residual = np.linalg.norm(shifts - coefficient * locations - translation, axis=1)
        median = float(np.median(residual[use]))
        use = residual <= max(0.35, median * 2.5)
    p, d, weight = locations[use], shifts[use], weights[use]
    error = float(np.sqrt(np.average(residual[use] ** 2, weights=weight)))
    initial = float(np.sqrt(np.average(np.sum(d ** 2, axis=1), weights=weight)))
    extent = np.ptp(p, axis=0)
    edge_shift = abs(coefficient) * math.hypot(w / 2, h / 2)
    accepted = (len(p) >= 10 and min(extent / [w, h]) >= 0.4
                and abs(coefficient) <= 0.012 and edge_shift <= 8
                and np.linalg.norm(translation) <= 0.75
                and error <= 0.45 and initial > 0.12
                and error < initial * 0.6 and edge_shift >= 0.15)
    return dict(status="applied" if accepted else "unchanged", tiles=len(p),
                scale=1.0 + coefficient if accepted else 1.0,
                fitted_scale=1.0 + coefficient, residual_pixels=error,
                initial_shift_rms=initial, translation_xy=translation.tolist(),
                edge_shift_pixels=edge_shift)


def apply_radial_scales(source, scales, *, center=None):
    """Keep the canvas, green samples and original out-of-support edges."""
    h, w = source.shape[:2]
    center = center or ((w - 1) / 2, (h - 1) / 2)
    result = source.copy()
    x = np.arange(w, dtype=np.float32)[None, :]
    for channel in (0, 2):
        scale = float(scales[channel])
        if scale == 1.0:
            continue
        plane = np.ascontiguousarray(source[..., channel])
        mx = center[0] + (x - center[0]) * scale
        for start in range(0, h, 256):
            stop = min(start + 256, h)
            y = np.arange(start, stop, dtype=np.float32)[:, None]
            my = center[1] + (y - center[1]) * scale
            mapx = np.broadcast_to(mx, (stop - start, w)).copy()
            mapy = np.broadcast_to(my, (stop - start, w)).copy()
            warped = cv2.remap(plane, mapx, mapy, cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
            valid = (mapx >= 2) & (mapx <= w - 3) & (mapy >= 2) & (mapy <= h - 3)
            result[start:stop, :, channel] = np.where(valid, warped, plane[start:stop])
    return np.ascontiguousarray(result)


def correct_chromatic_aberration(source, *, center=None):
    """Return corrected float32 pixels and an explicit acceptance receipt."""
    if (not isinstance(source, np.ndarray) or source.dtype != np.float32
            or source.ndim != 3 or source.shape[-1] != 3 or not np.isfinite(source).all()):
        raise ValueError("色差矫正需要有限的线性 float32 RGB")
    h, w = source.shape[:2]
    center = center or ((w - 1) / 2, (h - 1) / 2)
    receipt = dict(version=1, policy=POLICY, method="image-radial-estimation",
                   green_unchanged=True, canvas_unchanged=True, channels={})
    if min(h, w) < 192:
        receipt.update(status="insufficient", reason="画面过小，未应用色差估算")
        return source, receipt
    # Tile calculations use at most 1800 pixels on the long edge. This is
    # estimation only: the accepted transform is applied to the original.
    factor = min(1.0, 1800 / max(h, w))
    size = (round(w * factor), round(h * factor))
    sample = cv2.resize(source, size, interpolation=cv2.INTER_AREA) if factor < 1 else source
    selected_center = (center[0] * size[0] / w, center[1] * size[1] / h)
    reference = _feature(sample[..., 1])
    scales = [1.0, 1.0, 1.0]
    for channel in (0, 2):
        fit = _fit(reference, _feature(sample[..., channel]), selected_center)
        receipt["channels"]["RGB"[channel]] = fit
        scales[channel] = fit["scale"]
    changed = any(scale != 1.0 for scale in scales)
    receipt.update(status="applied" if changed else "unchanged", scales=scales,
                   center_xy=list(center), reason="" if changed else "未检测到可信的横向色差")
    return (apply_radial_scales(source, scales, center=center) if changed else source), receipt
