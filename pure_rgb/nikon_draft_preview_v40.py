"""Exact sampled Nikon CMS previews from an authenticated neutral Native TIFF.

The enclosing DraftPreview authenticates its parent manifest and RAW lifetime.
This adapter separately proves that the Native result is the complete neutral
reference required by Nikon, verifies the saved TIFF, and performs only the
requested output grid. It does not reinterpret a creative Native result, undo
an old adjustment, or replace the complete saving pipeline.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .common_positive import apply_common_positive
from .output_orientation import sample_oriented
from .output_orientation_cache import cache_budget_bytes
from .positive_lut import fit_point_codes
from .positive_post import post_codes
from .preview_data import verified_pixels
from .preview_grid import post_requires_full_grid
from .professional_controls import output_codes
from .workspace import checked_recipe, output_rectangle, source_recipe_identity


MODE = "cool-shared-rgb"
PREVIEW_LONG_EDGE = 1400
_NEUTRAL_NATIVE_KEYS = (
    "native_controls", "native_film", "native_float_controls",
    "native_spatial_controls", "highlight_controls", "common_positive_controls",
)


def _canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _readonly(pixels):
    return np.frombuffer(pixels.tobytes(), dtype=np.uint16).reshape(pixels.shape)


@dataclass(frozen=True)
class NikonPreviewResult:
    pixels: np.ndarray
    reference: np.ndarray
    icc: bytes
    sample_step: int
    receipt: dict
    reference_label: str = "原生中性参考"


class NikonDraftPreview:
    """One bounded sampled source and one bounded actual CMS result cache."""

    def __init__(self, manifest, parent_digest):
        if (not isinstance(parent_digest, str) or len(parent_digest) != 64
                or any(value not in "0123456789abcdef" for value in parent_digest)):
            raise ValueError("Nikon 快速预览需要已认证父清单SHA256")
        self.manifest = deepcopy(manifest)
        self.parent_digest = parent_digest
        self.saved_recipe = checked_recipe(self.manifest["recipe"])
        self.reference_identity = None
        self.sample = self.cms = self.coarse_cms = None
        self.sources_checked = False
        self.source_reads = 0

    def eligible(self, recipe):
        """Return false for unsupported producers, rather than guessing neutrality."""
        saved = self.saved_recipe
        if (
            self.manifest.get("state") != "complete"
            or saved.get("source_mode") == "bw-single-v1"
            or recipe.get("source_mode") == "bw-single-v1"
            # Shared masked measurement changes the neutral producer. Until
            # this adapter replays that frozen producer itself, use the full
            # authenticated color anchor; never reinterpret an old TIFF.
            or saved.get("density_metering")
            or recipe.get("density_metering")
            or not recipe.get("nikon_research_controls")
            or "native" not in self.manifest.get("results", {})
            or any(saved.get(key) for key in _NEUTRAL_NATIVE_KEYS)
            or saved["exposure_ev"] != 0
            or saved.get("output_controls", {}).get("native")
            or saved.get("post_controls", {}).get("native")
            or saved.get("output_orientation")
        ):
            return False
        if source_recipe_identity(recipe) != source_recipe_identity(saved):
            return False
        saved_roi = saved.get("roi") or self.manifest["roi"]
        requested_roi = recipe.get("roi") or self.manifest["roi"]
        if (
            requested_roi != saved_roi
            or recipe["density"] != saved["density"]
            or recipe.get("measurement_method") != saved.get("measurement_method")
            or recipe["paper"] != saved["paper"]
            or post_requires_full_grid(recipe.get("post_controls", {}).get(MODE))
        ):
            return False
        shape = self.manifest["source_shape"]
        return output_rectangle(saved, self.manifest["roi"], shape) == [
            0, 0, shape[1], shape[0]
        ]

    @property
    def retained_bytes(self):
        return sum(entry["pixels"].nbytes for entry in (self.sample, self.cms, self.coarse_cms) if entry)

    def close(self):
        self.sample = self.cms = self.coarse_cms = None

    def _source(self, recipe, full, sample_step_override=None):
        record = self.manifest["results"]["native"]
        identity = _identity(record["path"])
        if self.reference_identity is not None and identity != self.reference_identity:
            raise ValueError("Nikon 原生参考成片在认证后发生变化")
        shape = self.manifest["source_shape"]
        rectangle = output_rectangle(recipe, self.manifest["roi"], shape)
        x0, y0, x1, y1 = rectangle
        step = (sample_step_override if sample_step_override is not None else
                1 if full else max(1, int(np.ceil(max(x1 - x0, y1 - y0) / PREVIEW_LONG_EDGE))))
        direction = recipe.get("output_orientation")
        key = _canonical([rectangle, direction, step])
        if self.sample is not None and self.sample["key"] == key:
            return self.sample["pixels"], self.sample["icc"], step, rectangle
        pixels, icc = verified_pixels(record)
        self.source_reads += 1
        if (
            list(pixels.shape) != shape
            or hashlib.sha256(icc).hexdigest() != self.manifest["icc_sha256"]
            or _identity(record["path"]) != identity
        ):
            raise ValueError("Nikon 原生参考的完整网格、ICC或身份不一致")
        self.reference_identity = identity
        sample = sample_oriented(pixels[y0:y1, x0:x1], None, direction, step)
        self.sample = self.cms = self.coarse_cms = None
        if sample.nbytes * 2 <= cache_budget_bytes():
            sample = _readonly(sample)
            self.sample = dict(key=key, pixels=sample, icc=icc)
        else:
            sample.setflags(write=False)
        return sample, icc, step, rectangle

    def render(self, recipe, *, full=False, sample_step_override=None, interactive_edge=None):
        """Return an exact current-grid preview, or None for the normal fallback."""
        recipe = checked_recipe(recipe)
        if type(full) is not bool:
            raise ValueError("Nikon 预览分辨率必须为布尔值")
        if sample_step_override is not None and (type(sample_step_override) is not int or sample_step_override < 1):
            raise ValueError("Nikon 精细预览步长必须为正整数")
        if interactive_edge is not None and (full or sample_step_override is not None
                or type(interactive_edge) is not int or interactive_edge < 1):
            raise ValueError("Nikon 拖动预览尺寸只适用于适应窗口")
        if not self.eligible(recipe):
            return None
        if not self.sources_checked:
            from .engines import check_reference_sources
            from .native import check_baseline

            check_baseline()
            check_reference_sources()
            self.sources_checked = True
        from negpy.domain.models import WorkspaceConfig
        from .workflow_render import _nikon_reference_config, _nikon_reference_guard
        from .nikon_research_workflow import (
            checked_nikon_resources, render_nikon_research_workflow,
        )

        config = _nikon_reference_config(self.manifest)
        _nikon_reference_guard(WorkspaceConfig.from_flat_dict(deepcopy(config)))
        resources = checked_nikon_resources()
        previous_reads = self.source_reads
        reference, icc, step, rectangle = self._source(recipe, full, sample_step_override)
        slot = "cms"
        if interactive_edge is not None:
            factor = (max(reference.shape[:2]) + interactive_edge - 1) // interactive_edge
            if factor > 1:
                # Reuse the authenticated, already-oriented fit grid. Taking
                # every nth point preserves its phase without reopening TIFF.
                reference = _readonly(reference[::factor, ::factor])
                step *= factor
                slot = "coarse_cms"
        document = recipe["nikon_research_controls"]
        key = _canonical([self.sample["key"] if self.sample else None,
                          list(reference.shape), rectangle,
                          recipe.get("output_orientation"), step, document,
                          recipe.get("experimental_film_controls")])
        cached = getattr(self, slot)
        cms_cache_hit = cached is not None and cached["key"] == key
        if cms_cache_hit:
            converted, cms_receipt = cached["pixels"], cached["receipt"]
        else:
            source_key = hashlib.sha256(_canonical({
                "parent_manifest_sha256": self.parent_digest,
                "native_TIFF_sha256": self.manifest["results"]["native"]["sha256"],
                "reference_config": config,
            }).encode()).hexdigest()
            # The selected oriented sample is declared as its own RGB16 input.
            # Fixed LUT, CML4 and ICC are point operators, so their output is
            # exactly full conversion then the same orientation/crop/stride.
            converted, other, cms_receipt = render_nikon_research_workflow(
                reference, document, working_icc=icc,
                source_key=source_key, reference_config=config,
                experimental_film=recipe.get("experimental_film_controls"),
            )
            if other != icc:
                raise ValueError("Nikon 快速预览工作ICC不一致")
            setattr(self, slot, None)
            if (self.retained_bytes + converted.nbytes) <= cache_budget_bytes():
                converted = _readonly(converted)
                setattr(self, slot, dict(key=key, pixels=converted, receipt=cms_receipt))
        output = recipe.get("output_controls", {}).get(MODE) or dict(version=1, values={})
        post = recipe.get("post_controls", {}).get(MODE)
        positive = apply_common_positive(converted, recipe.get("common_positive_controls"))
        current = (fit_point_codes(positive, output, recipe["exposure_ev"], post)
                   if not full and sample_step_override is None else None)
        if current is None:
            current, _ = output_codes(positive, output, recipe["exposure_ev"])
            current, _ = post_codes(current, post, sample_step=step, collect_ranges=False)
        record = self.manifest["results"]["native"]
        if (
            _identity(record["path"]) != self.reference_identity
            or checked_nikon_resources() != resources
        ):
            raise ValueError("Nikon 预览期间参考或资源发生变化")
        receipt = dict(
            path="authenticated-neutral-native-point-preview",
            source_TIFF_sha256=record["sha256"], source_shape=self.manifest["source_shape"],
            source_roi=rectangle, output_orientation=recipe.get("output_orientation"),
            sample_step=step, sample_shape=list(reference.shape),
            CMS_input_pixels=int(reference.shape[0] * reference.shape[1]),
            actual_CMS_pixels=(0 if cms_cache_hit else int(reference.shape[0] * reference.shape[1])),
            CMS_evaluate_calls=0 if cms_cache_hit else 1,
            CMS_cache_hit=cms_cache_hit,
            source_TIFF_reads=self.source_reads - previous_reads,
            RAW_decode_calls=0, Native_render_calls=0, legacy_fit_calls=0,
            exact_full_grid_sample=True,
            CMS_binding_sha256=cms_receipt["binding_sha256"],
        )
        return NikonPreviewResult(
            np.array(current, copy=True), np.array(reference, copy=True), icc, step, receipt
        )
