"""Explicit pre-paper shoulder for Native normalized negative log fractions.

This experimental CPU hook repositions existing dense-negative values before
the original photographic-paper curve. It does not reconstruct clipped RAW,
invert a bounded TIFF, estimate new meters, or certify recovered scene colour.
"""

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import RenderIntent
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.process.models import ProcessMode
from negpy.services.rendering.engine import DarkroomEngine


DOMAIN = "native-normalized-negative-log-fraction"
SCHEMA = "pure-rgb-native-highlight-recovery"
VERSION = 1
KNEE = 0.25
BASE_METRIC_KEYS = (
    "log_bounds_base",
    "pre_trim_bounds",
    "final_bounds",
    "norm_density_range",
    "metered_anchor",
    "neutral_axis_refs",
    "textural_range",
    "shadow_point",
    "highlight_point",
)
_METRIC_KEY = "pure_rgb_highlight_recovery_v1"
_LIMITS = [
    "Only max(qRGB)<0.25 is mapped: all three channels must lie on the dense-negative side.",
    "Single-channel highlights and saturated colours may remain unchanged.",
    "Existing finite source distinctions may be repositioned; missing or clipped source information is not reconstructed.",
    "Normalized channel differences are shared; downstream nonlinear paper and colour operations can still change colour.",
    "Original paper, clipping, output encoding and quantization remain bounded.",
]


def checked_highlight_recovery(doc=None):
    """Return an independent, finite, versioned document; absent means disabled."""
    if doc is None:
        doc = {}
    if not isinstance(doc, dict) or set(doc) - {"version", "strength"}:
        raise ValueError("Highlight recovery expects only version and strength")
    version = doc.get("version", VERSION)
    strength = doc.get("strength", 0.0)
    if type(version) is not int or version != VERSION:
        raise ValueError("Unsupported highlight recovery version")
    if (
        type(strength) not in (int, float)
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("Highlight recovery strength must be finite in [0,1]")
    return {"version": VERSION, "strength": float(strength)}


def _image(q):
    if (
        not isinstance(q, np.ndarray)
        or q.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or q.ndim != 3
        or q.shape[-1] != 3
        or min(q.shape[:2], default=0) < 1
        or not q.flags.c_contiguous
        or not np.isfinite(q).all()
    ):
        raise ValueError("Expected finite C-contiguous float32/64 HWC RGB q")


def _range(image):
    return {
        "minimum_rgb": image.min(axis=(0, 1)).tolist(),
        "maximum_rgb": image.max(axis=(0, 1)).tolist(),
    }


def map_negative_highlights(q, doc=None, *, domain=DOMAIN):
    """Map a declared Native q buffer without changing its source or its meters.

    Native q=(log10(T)-floor)/(ceil-floor), so low/negative q is the dense
    negative / bright-positive side. This domain is not positive scene density.
    """
    document = checked_highlight_recovery(doc)
    if domain != DOMAIN:
        raise ValueError("Highlight recovery requires the declared Native q domain")
    _image(q)
    strength = document["strength"]
    count = int(q.shape[0] * q.shape[1])
    receipt = {
        "schema": SCHEMA,
        **document,
        "domain": DOMAIN,
        "knee": KNEE,
        "dtype": q.dtype.str,
        "shape": list(q.shape),
        "pixel_count": count,
        "input_range": _range(q),
        "quality_limits": deepcopy(_LIMITS),
    }
    if strength == 0:
        receipt.update(
            status="disabled",
            affected_pixel_count=0,
            affected_fraction=0.0,
            identity=True,
            output_range=deepcopy(receipt["input_range"]),
        )
        return q, receipt

    maximum = np.max(q.astype(np.float64, copy=False), axis=-1)
    selected = maximum < KNEE
    affected = int(np.count_nonzero(selected))
    output = q.copy()
    m = maximum[selected]
    d = KNEE - m
    # Equivalent to the specified rational shoulder, without d/K overflow for
    # large finite negative q. Subtract m before adding mNew to avoid q+delta
    # cancellation near the full-strength asymptote.
    m_new = (KNEE / (KNEE + strength * d)) * ((1.0 - strength) * m + strength * KNEE)
    output[selected] = (
        q[selected].astype(np.float64) - m[:, None] + m_new[:, None]
    ).astype(q.dtype)
    if not np.isfinite(output).all():
        raise ValueError("Highlight shoulder produced non-finite q")
    receipt.update(
        status="mapped",
        affected_pixel_count=affected,
        affected_fraction=affected / count,
        identity=bool(np.array_equal(q, output)),
        output_range=_range(output),
        unaffected_pixels_exact=True,
    )
    return output, receipt


def _json_metric(value):
    if isinstance(value, LogNegativeBounds):
        return {
            "floors": _json_metric(value.floors),
            "ceils": _json_metric(value.ceils),
        }
    if isinstance(value, np.ndarray):
        return _json_metric(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_metric(v) for v in value]
    if isinstance(value, np.generic):
        return _json_metric(value.item())
    if isinstance(value, dict):
        return {str(k): _json_metric(v) for k, v in value.items()}
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _supported(settings):
    if not isinstance(settings, WorkspaceConfig):
        raise ValueError("Expected a Native WorkspaceConfig")
    p, e, g, r = (
        settings.process,
        settings.exposure,
        settings.geometry,
        settings.retouch,
    )
    if (
        p.process_mode != ProcessMode.C41
        or p.positive_source
        or e.render_intent != RenderIntent.PRINT
    ):
        raise ValueError("Highlight recovery v1 requires Native C41 negative PRINT CPU")
    if (
        g.rotation != 0
        or g.fine_rotation != 0
        or g.flip_horizontal
        or g.flip_vertical
        or g.converge_h != 0
        or g.converge_v != 0
        or g.distortion_k1 != 0
        or g.lens_distortion_from_metadata
        or g.lens_ca_from_metadata
        or (g.crop_from_auto and g.crop_rect is None)
    ):
        raise ValueError(
            "Highlight recovery v1 requires resolved axis-aligned geometry"
        )
    if (
        r.dust_remove
        or r.ir_dust_remove
        or r.manual_dust_spots
        or r.manual_heal_strokes
        or r.scratch_lines
        or r.clone_strokes
        or r.dust_exclusion_strokes
    ):
        raise ValueError("Highlight recovery v1 does not support source retouch")
    if settings.local.masks or e.contrast_mask_strength != 0 or e.diffusion != 0:
        raise ValueError(
            "Highlight recovery v1 does not support local/contrast/diffusion masks"
        )


class HighlightRecoveryEngine(DarkroomEngine):
    """Original CPU graph with one explicit q hook before exposure/paper.

    The document is instance-bound. Disabled instances use the original stage
    callbacks and cache keys verbatim; observing summaries changes no metrics.
    Ordinary original Lab controls are retained, including spatial effects.
    """

    def __init__(self, doc=None):
        super().__init__()
        self._document = checked_highlight_recovery(doc)
        self._last_receipt = None
        self._last_base_metrics = None
        self._last_base_metrics_sha256 = None
        self._source_hash = None
        self._spatial = None

    @property
    def document(self):
        return deepcopy(self._document)

    @property
    def last_receipt(self):
        return deepcopy(self._last_receipt)

    @property
    def last_stage_receipt(self):
        return self.last_receipt

    @property
    def last_base_metrics(self):
        return deepcopy(self._last_base_metrics)

    @property
    def last_base_metrics_sha256(self):
        return self._last_base_metrics_sha256

    def process(self, img, settings, source_hash, context=None):
        if self._document["strength"] != 0:
            _supported(settings)
        self._source_hash = source_hash
        self._last_receipt = None
        self._last_base_metrics = None
        self._last_base_metrics_sha256 = None
        self._spatial = {
            key: float(getattr(settings.lab, key))
            for key in (
                "clahe_strength",
                "sharpen",
                "chroma_denoise",
                "glow_amount",
                "halation_strength",
            )
        }
        return super().process(img, settings, source_hash, context)

    def _run_stage(
        self, img, config, cache_field, processor_fn, context, pipeline_changed
    ):
        strength = self._document["strength"]
        if cache_field == "exposure":
            self._last_base_metrics = {
                key: _json_metric(context.metrics.get(key)) for key in BASE_METRIC_KEYS
            }
            self._last_base_metrics_sha256 = _digest(self._last_base_metrics)
            if strength != 0:
                original = processor_fn

                def processor_fn(image, ctx):
                    mapped, receipt = map_negative_highlights(image, self._document)
                    ctx.metrics[_METRIC_KEY] = receipt
                    return original(mapped, ctx)

                config = (config, SCHEMA, VERSION, strength)

        output, changed = super()._run_stage(
            img, config, cache_field, processor_fn, context, pipeline_changed
        )
        if cache_field == "exposure":
            if strength == 0:
                receipt = {
                    "schema": SCHEMA,
                    **self.document,
                    "status": "disabled",
                    "domain": DOMAIN,
                    "knee": KNEE,
                    "identity": True,
                    "affected_pixel_count": 0,
                    "affected_fraction": 0.0,
                    "stage_hook_installed": False,
                    "quality_limits": deepcopy(_LIMITS),
                }
            else:
                receipt = deepcopy(context.metrics[_METRIC_KEY])
                receipt["stage_hook_installed"] = True
            receipt.update(
                source_hash=self._source_hash,
                base_metrics_sha256=self._last_base_metrics_sha256,
                original_meters_unchanged=True,
                downstream_lab_spatial_controls=deepcopy(self._spatial),
                downstream_spatial_may_affect_neighbours=any(self._spatial.values()),
            )
            self._last_receipt = receipt
        return output, changed
