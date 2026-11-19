"""Exact sampled Flex research, before any output-only finishing.

The caller owns RAW/file identity and full-grid frozen metering validation.
This helper never measures a smaller source or reconstructs transmission from
a Native TIFF. It samples the already authenticated source grid, then runs the
same pixel-local Flex density/LUT/ColorMap/ICC implementation used by saves.
Neighbourhood/geometry finishing must instead request sample_step=1 and take
the sole final display stride after that finishing stage.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np

from .engines import Analysis
from .flex_research_workflow import (
    _geometry,
    _sha,
    _source_layout,
    checked_flex_research_controls,
    render_flex_research_workflow,
)
from .tone_study import ICC_SHA256

MODE = "flex-endpoint"
PREVIEW_LONG_EDGE = 1400


def _canonical(document):
    return json.dumps(document, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _prepare(source, analysis, document, working_icc, source_key, view_roi,
             sample_step, experimental_film=None):
    _source_layout(source)
    if not isinstance(analysis, Analysis):
        raise ValueError("Flex sampled preview requires the full-grid frozen Analysis")
    _sha(source_key)
    if not isinstance(working_icc, bytes) or hashlib.sha256(working_icc).hexdigest() != ICC_SHA256:
        raise ValueError("Flex sampled preview requires the actual working ICC")
    controls = checked_flex_research_controls(document)
    if controls is None:
        raise ValueError("Flex sampled preview requires active research controls")
    if type(sample_step) is not int or sample_step < 1:
        raise ValueError("Flex sampled preview step must be a positive integer")
    height, width = source.shape[:2]
    roi = [0, 0, width, height] if view_roi is None else list(view_roi)
    if len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError("Flex sampled preview ROI must contain four integers")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Flex sampled preview ROI is outside the authenticated source")
    # Own only the requested grid. No 96MP float64 copy, Native export or
    # legacy endpoint fit is necessary for this explicit Flex route.
    sample = np.array(source[y0:y1:sample_step, x0:x1:sample_step], copy=True, order="C")
    sample.setflags(write=False)
    geometry = _geometry(
        dict(full_source_shape=list(source.shape), view_roi=roi, sample_step=sample_step),
        sample.shape,
    )
    analysis_doc = dict(
        dark=list(analysis.dark), base=list(analysis.base), roi=list(analysis.roi),
        percentiles=list(analysis.percentiles) if analysis.percentiles is not None else None,
    )
    if not np.isfinite(sample).all():
        raise ValueError("Nonfinite sample rejected before Flex/ICC processing")
    sample_sha = hashlib.sha256(memoryview(sample).cast("B")).hexdigest()
    binding = dict(
        controls=controls, analysis=analysis_doc, source_key=source_key,
        source_geometry=geometry, sampled_source_sha256=sample_sha,
        sampled_source_dtype=sample.dtype.str, working_icc_sha256=ICC_SHA256,
        experimental_film=experimental_film,
    )
    key = hashlib.sha256(_canonical(binding).encode()).hexdigest()
    return sample, geometry, controls, binding, key


class FlexDraftSampleCache:
    """One bounded, immutable color sample; source RAW remains caller-owned."""

    def __init__(self, budget_bytes=16 * 1024**2):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("Flex sample cache budget must be a nonnegative integer")
        self.budget_bytes = budget_bytes
        self.entry = None
        self.retained_bytes = 0

    def clear(self):
        self.entry = None
        self.retained_bytes = 0

    def render(
        self, source, analysis, document, *, working_icc, source_key,
        view_roi=None, sample_step=1, experimental_film=None,
    ):
        sample, geometry, controls, binding, key = _prepare(
            source, analysis, document, working_icc, source_key, view_roi,
            sample_step, experimental_film=experimental_film,
        )
        # _prepare strictly rechecks resource bytes even for a hot result.
        if self.entry is not None and self.entry["key"] == key:
            receipt = deepcopy(self.entry["receipt"])
            receipt["preview"]["cache_hit"] = True
            return self.entry["pixels"].copy(), working_icc, receipt
        output, icc, receipt = render_flex_research_workflow(
            sample, analysis, controls, working_icc=working_icc,
            source_key=source_key, source_geometry=geometry,
            experimental_film=experimental_film,
        )
        receipt["preview"] = dict(
            method="full-grid frozen Analysis then exact source-grid stride before pixel-local Flex",
            binding=binding, cache_hit=False,
            full_source_pixels=int(source.shape[0] * source.shape[1]),
            processed_pixels=int(sample.shape[0] * sample.shape[1]),
            full_source_remeasured=False, legacy_endpoint_fit_applied=False,
            native_render_calls=0, output_finishing_applied=False,
            source_integrity_scope="Caller authenticates full source; this helper hashes the requested sample",
        )
        self.clear()
        if output.nbytes <= self.budget_bytes:
            retained = np.frombuffer(output.tobytes(), dtype=np.uint16).reshape(output.shape)
            self.entry = dict(key=key, pixels=retained, receipt=deepcopy(receipt))
            self.retained_bytes = retained.nbytes
        return output, icc, receipt


def render_flex_draft_sample(
    source, analysis, document, *, working_icc, source_key,
    view_roi=None, sample_step=1, experimental_film=None,
):
    """Stateless equivalent; use FlexDraftSampleCache for repeated UI edits."""
    return FlexDraftSampleCache(budget_bytes=0).render(
        source, analysis, document, working_icc=working_icc, source_key=source_key,
        view_roi=view_roi, sample_step=sample_step,
        experimental_film=experimental_film,
    )


def _file_identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def verified_tiff_icc(record, expected_sha256=None):
    """Authenticate a saved TIFF and read its ICC tag without decoding pixels."""
    import tifffile
    from .native import sha256

    path = Path(record["path"])
    before = _file_identity(path)
    if sha256(path) != record["sha256"]:
        raise ValueError("Flex参考TIFF字节已改变")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if page.dtype != np.dtype("uint16") or len(page.shape) != 3 or page.shape[2] != 3:
            raise ValueError("Flex参考必须是RGB16 TIFF")
        if record.get("shape") is not None and list(page.shape) != list(record["shape"]):
            raise ValueError("Flex参考TIFF网格与保存记录不一致")
        tag = page.tags.get(34675)
        if tag is None:
            raise ValueError("Flex参考TIFF缺少实际ICC")
        icc = bytes(tag.value)
    digest = hashlib.sha256(icc).hexdigest()
    if digest != ICC_SHA256 or any(
        value is not None and digest != value
        for value in (expected_sha256, record.get("icc_sha256"))
    ):
        raise ValueError("Flex参考TIFF的实际ICC与冻结工作域不一致")
    if _file_identity(path) != before:
        raise ValueError("Flex参考TIFF在认证期间发生变化")
    return icc, before


def oriented_source_sample_roi(rectangle, direction, step):
    """The positive-stride source phase for full orientation then stride.

    For an input axis which runs backwards in output coordinates, the final
    sample includes the original last pixel, not the origin-aligned reduced
    grid's last pixel. Shifting that axis by (size-1)%step preserves odd edges.
    """
    from .output_orientation import _matrix

    matrix = _matrix(direction)
    x0, y0, x1, y1 = rectangle
    phases = [0, 0]
    for axis, size in enumerate((x1 - x0, y1 - y0)):
        signs = matrix[:, axis]
        sign = next(int(value) for value in signs if value)
        if sign < 0:
            phases[axis] = (size - 1) % step
    return [x0 + phases[0], y0 + phases[1], x1, y1]


class FlexDraftPreview:
    """Lazy active-Flex path for DraftPreview, including Native-only projects.

    The owner authenticates the manifest and RAW lifetime around try_render.
    This object shares its lazily loaded frozen source Session with the owner;
    it does not own or close that Session and never retains a full TIFF frame.
    """

    def __init__(self, owner):
        from .output_orientation_cache import cache_budget_bytes

        self.owner = owner
        # Two small caches share at most the owner's process retention budget.
        self.color_cache = FlexDraftSampleCache(min(16 * 1024**2, cache_budget_bytes() // 2))
        self.profile = None
        self.reference = None
        self.source_session = self.source_key = None

    def close(self):
        self.color_cache.clear()
        self.profile = self.reference = None
        self.source_session = self.source_key = None

    @property
    def retained_bytes(self):
        return self.color_cache.retained_bytes + (
            self.reference["pixels"].nbytes if self.reference is not None else 0
        )

    def _eligible(self, recipe):
        from .preview_grid import post_requires_full_grid

        saved, manifest = self.owner.recipe, self.owner.manifest
        if (
            not recipe.get("flex_research_controls")
            or recipe.get("source_mode") == "bw-single-v1"
            or saved.get("source_mode") == "bw-single-v1"
            # A shared mask must be replayed by the complete frozen-analysis
            # pipeline before sampling, not read from the initial rectangular
            # density_window in this legacy source-sample adapter.
            or recipe.get("density_metering")
            or saved.get("density_metering")
            or post_requires_full_grid(recipe.get("post_controls", {}).get(MODE))
        ):
            return False
        if any(recipe.get(key) != saved.get(key) for key in (
            "raw", "raw_import", "flatfield", "registration_check", "source_mode",
        )):
            return False
        saved_roi = saved.get("roi") or manifest["roi"]
        requested_roi = recipe.get("roi") or manifest["roi"]
        return (
            list(saved_roi) == list(requested_roi)
            and recipe["density"] == saved["density"]
            and recipe["paper"] == saved["paper"]
        )

    def _icc(self):
        manifest = self.owner.manifest
        records = manifest["results"]
        record = records.get("native") or records.get(MODE) or next(iter(records.values()))
        identity = _file_identity(record["path"])
        if self.profile is None:
            icc, identity = verified_tiff_icc(record, manifest.get("icc_sha256"))
            self.profile = dict(record=deepcopy(record), identity=identity, icc=icc)
        elif self.profile["record"] != record or self.profile["identity"] != identity:
            raise ValueError("Flex参考TIFF在预览认证后发生变化")
        return self.profile["icc"]

    def _source(self):
        from .live_worker import Session
        from .workflow_render import _export_source_key

        owner = self.owner
        if owner.saved_session is None:
            owner.saved_session = Session(owner.path, owner.digest, owner.directory)
        session = owner.saved_session
        session.prepare_raw()
        if list(session.source.shape) != list(owner.manifest["source_shape"]):
            raise ValueError("Flex预览来源与冻结完整源网格不一致")
        if self.source_session is None or self.source_session() is not session:
            self.source_key = _export_source_key(session.processor, session.config, owner.records)
            # Do not keep a full RAW session alive after the owner releases it
            # while switching to another analysis route.
            self.source_session = weakref.ref(session)
        owner.processor = session.processor
        return session.source, self.source_key

    def _reference(self, rectangle, direction, step, icc, fallback):
        from .output_orientation import orient_reference, sample_oriented
        from .preview_data import verified_pixels
        from .workspace import output_rectangle

        owner = self.owner
        record = owner.manifest["results"].get("native")
        if record is None or any(
            key.startswith("geometry.")
            for key in owner.recipe.get("post_controls", {}).get("native", {}).get("values", {})
        ):
            return fallback.copy(), "Flex草稿调整前"
        saved_roi = output_rectangle(owner.recipe, owner.manifest["roi"], owner.manifest["source_shape"])
        a, b, c, d = saved_roi
        x0, y0, x1, y1 = rectangle
        if not (a <= x0 < x1 <= c and b <= y0 < y1 <= d):
            return fallback.copy(), "Flex草稿调整前"
        identity = _file_identity(record["path"])
        key = _canonical([record, rectangle, direction, step])
        if self.reference is not None and self.reference["key"] == key:
            if identity != self.reference["identity"]:
                raise ValueError("Flex原生参考成片在认证后发生变化")
            return self.reference["pixels"].copy(), "已保存原生参考"
        pixels, actual_icc = verified_pixels(record)
        if actual_icc != icc:
            raise ValueError("Flex原生参考实际ICC与当前工作域不一致")
        pixels = orient_reference(pixels, owner.recipe.get("output_orientation"), None)
        if pixels.shape[:2] != (d - b, c - a):
            raise ValueError("Flex原生参考的裁切网格与保存配方不一致")
        view = pixels[y0 - b:y1 - b, x0 - a:x1 - a]
        reference = sample_oriented(view, None, direction, step)
        if _file_identity(record["path"]) != identity:
            raise ValueError("Flex原生参考在采样期间发生变化")
        self.reference = None
        if reference.nbytes <= self.color_cache.budget_bytes:
            immutable = np.frombuffer(reference.tobytes(), dtype=np.uint16).reshape(reference.shape)
            self.reference = dict(key=key, identity=identity, pixels=immutable)
        return reference, "已保存原生参考"

    def try_render(self, recipe, full=False, sample_step_override=None):
        from .output_orientation import orient_codes
        from .workspace import output_rectangle

        if type(full) is not bool:
            raise ValueError("Flex预览分辨率必须为布尔值")
        if sample_step_override is not None and (type(sample_step_override) is not int or sample_step_override < 1):
            raise ValueError("Flex精细预览步长必须为正整数")
        if not self._eligible(recipe):
            return None
        icc = self._icc()
        source, source_key = self._source()
        rectangle = output_rectangle(recipe, self.owner.manifest["roi"], source.shape)
        x0, y0, x1, y1 = rectangle
        step = (sample_step_override if sample_step_override is not None else
                1 if full else max(1, int(np.ceil(max(x1 - x0, y1 - y0) / PREVIEW_LONG_EDGE))))
        direction = recipe.get("output_orientation")
        phase_roi = oriented_source_sample_roi(rectangle, direction, step)
        analysis = Analysis(**self.owner.manifest["density_window"]["analysis"])
        pixels, actual_icc, receipt = self.color_cache.render(
            source, analysis, recipe["flex_research_controls"],
            working_icc=icc, source_key=source_key, view_roi=phase_roi, sample_step=step,
            experimental_film=recipe.get("experimental_film_controls"),
        )
        before = orient_codes(pixels, direction)
        # This branch excludes neighbourhood/geometry post. Point finishing
        # commutes with exact orthogonal pixel permutation and grid sampling.
        current = orient_codes(self.owner._finish(pixels, recipe, MODE), direction)
        reference, label = self._reference(rectangle, direction, step, icc, before)
        if current.shape != reference.shape or actual_icc != icc:
            raise ValueError("Flex快速预览／调整前几何或ICC不一致")
        receipt["preview"].update(
            requested_output_roi=rectangle, phase_source_roi=phase_roi,
            output_orientation=deepcopy(direction), reference_label=label,
            full_resolution=full,
        )
        return current, reference, icc, step, receipt
