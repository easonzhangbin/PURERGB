"""Explicit Native zonal/spatial save and replay, sharing the original encoder.

Other renderers keep their original neutral reference. A saved spatial packet
is restored before replay; a change of spatial strengths reuses its metering.
No UI, scanner calibration, RAW decoder or vendor mathematics is replaced.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from negpy.domain.models import (
    ColorSpace,
    ExportFormat,
    TiffCompression,
    WorkspaceConfig,
)
from negpy.features.lab.models import LabConfig, SharpenMethod
from negpy.features.local.models import LocalMask, MaskKey, MaskShape
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE

from .bw_metering import BWMaskedState, checked_bw_metering
from .frozen_native_cpu import FrozenNativeCPU, _config_document
from .metering_masks import build_metering_mask
from .metering_snapshot import (
    COORDINATE_SPACE,
    SOURCE_DOMAIN,
    _check_source,
    freeze_zonal_metering,
    restore_zonal_metering,
)
from .native_spatial_v30 import (
    LAB_FIELDS,
    LAB_RANGES,
    NativeSpatialCPU,
    _local,
    _render_configuration,
)


SCHEMA = "pure-rgb-native-spatial-workflow-v31"
MAX_BYTES = 32 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_highlight_reference_bounds(reference, source_shape):
    """Validate a manual reference in source coordinates, independently of metering."""
    x0, y0, x1, y1 = reference
    _require(
        0 <= x0 < x1 <= source_shape[1]
        and 0 <= y0 < y1 <= source_shape[0]
        and min(x1 - x0, y1 - y0) >= 4,
        "高光参考区必须位于原图内，且至少为 4 × 4 源像素",
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pixels(value):
    from .immutable_source_v44 import source_digest

    return source_digest(value)


def _runtime():
    names = (
        "spatial_workflow_v31.py",
        "native_spatial_v30.py",
        "automatic_metering_v31.py",
        "adaptive_edge_v31.py",
        "defect_reliability_v32.py",
        "film_density_domain_v45.py",
    )
    return {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in names
    }


def checked_spatial_controls(document):
    """Strict metadata; OFF normalizes away to the unchanged existing route."""
    _require(type(document) is dict, "Spatial controls require a versioned object")
    _require(
        type(document.get("version")) is int and document["version"] == 1,
        "Unsupported spatial workflow version",
    )
    _require(type(document.get("enabled")) is bool, "Spatial enabled must be boolean")
    if not document["enabled"]:
        _require(
            set(document) == {"version", "enabled"},
            "Disabled spatial controls cannot hide fields",
        )
        return None
    _require(
        set(document)
        <= {
            "version",
            "enabled",
            "metering",
            "lab",
            "local_masks",
            "automatic_avoidance",
        },
        "Unknown spatial workflow field",
    )
    _require(
        len(_canonical(document).encode()) <= 512 * 1024,
        "Spatial controls exceed metadata bound",
    )
    meter = checked_bw_metering(document.get("metering"))
    _require(
        meter is not None, "Spatial workflow requires explicit active zonal metering"
    )
    lab = document.get("lab", {})
    _require(
        type(lab) is dict and not set(lab) - LAB_FIELDS, "Unknown spatial Lab field"
    )
    lab = deepcopy(lab)
    for name, value in lab.items():
        if name in LAB_RANGES:
            lo, hi = LAB_RANGES[name]
            _require(
                type(value) in (int, float)
                and np.isfinite(value)
                and lo <= value <= hi,
                "Spatial Lab value outside supported range",
            )
        elif name == "sharpen_auto_masking":
            _require(value is False, "Spatial auto-mask sharpening is not supported")
        else:
            _require(
                type(value) is str and value in {x.value for x in SharpenMethod},
                "Unknown sharpening method",
            )
    masks = document.get("local_masks", [])
    _require(type(masks) is list and len(masks) <= 64, "At most64 spatial local masks")
    normalized = []
    for mask in masks:
        _require(
            type(mask) is dict
            and not set(mask)
            - {"shape", "vertices", "stops", "feather", "invert", "enabled"},
            "Unknown local exposure mask field",
        )
        _require(
            "vertices" in mask and type(mask["vertices"]) is list,
            "Local mask needs explicit vertices",
        )
        normalized.append(deepcopy(mask))
    result = dict(
        version=1, enabled=True, metering=meter, lab=lab, local_masks=normalized
    )
    if "automatic_avoidance" in document:
        from .automatic_metering_v31 import checked_automatic_avoidance

        automatic = checked_automatic_avoidance(document["automatic_avoidance"])
        if automatic is not None:
            result["automatic_avoidance"] = automatic
    # Validate vector degeneracy, inactive fields and ranges without a source.
    _render_config(WorkspaceConfig(), result)
    return result


def check_spatial_recipe(recipe):
    value = checked_spatial_controls(recipe["native_spatial_controls"])
    if value is None:
        recipe.pop("native_spatial_controls")
        return
    controls = recipe.get("native_controls", {}).get("values", {})
    _require(
        not any(key.startswith("lab.") and key[4:] in LAB_FIELDS for key in controls),
        "Place spatial Lab settings in native_spatial_controls, without duplicate Native overrides",
    )
    if recipe.get("bw_metering"):
        _require(
            recipe["bw_metering"] == value["metering"],
            "BW and spatial metering selections must match",
        )
    recipe["native_spatial_controls"] = value


def _meter_config(config):
    default = LabConfig()
    lab = {name: getattr(default, name) for name in LAB_FIELDS}
    if config.process.process_mode == ProcessMode.C41:
        lab.update(
            sharpen=0,
            chroma_denoise=0,
            clahe_strength=0,
            glow_amount=0,
            halation_strength=0,
        )
    return replace(
        config, lab=replace(config.lab, **lab), local=replace(config.local, masks=())
    )


def _render_config(config, document):
    lab = deepcopy(document["lab"])
    if "sharpen_method" in lab:
        lab["sharpen_method"] = SharpenMethod(lab["sharpen_method"])
    masks = []
    for item in document["local_masks"]:
        value = dict(item)
        value["shape"] = MaskShape(value.get("shape", MaskShape.POLYGON.value))
        value["vertices"] = tuple(tuple(p) for p in value["vertices"])
        value["key"] = MaskKey.OFF
        masks.append(LocalMask(**value))
    result = replace(
        config,
        lab=replace(config.lab, **lab),
        local=replace(config.local, masks=tuple(masks)),
    )
    _local(result.local)
    return result


def _identity(source, source_key):
    from .film_density_domain_v45 import DOMAIN, film_domain_for

    _require(
        type(source_key) is str
        and len(source_key) == 64
        and all(c in "0123456789abcdef" for c in source_key),
        "Spatial source key requires SHA256",
    )
    return dict(
        policy=SCHEMA,
        shape=list(source.shape),
        dtype=source.dtype.str,
        source_pixels_sha256=_pixels(source),
        geometry_token=_digest(dict(source_key=source_key, shape=list(source.shape))),
        coordinate_space=COORDINATE_SPACE,
        domain=DOMAIN if film_domain_for(source) is not None else SOURCE_DOMAIN,
        channel_order="RGB",
        source_read_only=True,
    )


def _support(source, roi, document):
    meter = document["metering"]
    return build_metering_mask(
        source.shape, roi, include=meter["include"], exclude=meter["exclude"]
    )


def _c41_packet(source, config, roi, document, identity, *, checkpoint=None):
    from .immutable_source_v44 import certificate_for
    from .metering_snapshot import ALGORITHM

    meter = document["metering"]

    def freeze():
        return freeze_zonal_metering(
            source, identity, roi, meter["excluded_rois"],
            zones=tuple(meter["zones"]), blocks_per_zone=tuple(meter["blocks_per_zone"]),
            padding=meter["padding"], min_area_fraction=meter["min_area_fraction"],
            selection_mask=_support(source, roi, document),
        )

    certificate = certificate_for(source)
    bundle = (certificate.density_bundle(roi, meter, identity, freeze, policy=ALGORITHM)
              if certificate is not None else freeze())
    if checkpoint is not None:
        checkpoint("density_bounds")
    packet = FrozenNativeCPU(source, config, bundle.to_json(), identity)
    return packet, bundle


def _project_automatic_support(source, roi, document, automatic):
    """Budget whole metering leaves, including padding and explicit protection."""
    from .zonal_metering import _edges, _intersects

    boxes = automatic["automatic_exclusions"]
    if not boxes:
        return
    meter = document["metering"]
    selection = _support(source, roi, document).mask
    x0, y0, x1, y1 = roi
    padding = meter["padding"]

    def expand(box):
        a, b, c, d = box
        return (
            max(x0, a - padding),
            max(y0, b - padding),
            min(x1, c + padding),
            min(y1, d + padding),
        )

    manual = [expand(box) for box in meter["excluded_rois"]]
    proposed = [expand(box) for box in boxes]
    zr, zc = meter["zones"]
    br, bc = meter["blocks_per_zone"]
    ys, xs = _edges(y0, y1, zr), _edges(x0, x1, zc)
    available = removed = 0
    touches_protected = False
    for iy in range(zr):
        for ix in range(zc):
            sy, sx = _edges(ys[iy], ys[iy + 1], br), _edges(xs[ix], xs[ix + 1], bc)
            for py in range(br):
                for px in range(bc):
                    leaf = (sx[px], sy[py], sx[px + 1], sy[py + 1])
                    if any(_intersects(leaf, box) for box in manual):
                        continue
                    area = int(
                        selection[
                            leaf[1] - y0 : leaf[3] - y0, leaf[0] - x0 : leaf[2] - x0
                        ].sum()
                    )
                    available += area
                    if any(_intersects(leaf, box) for box in proposed):
                        removed += area
                        touches_protected |= any(
                            _intersects(leaf, box)
                            for box in document["automatic_avoidance"]["protected_rois"]
                        )
    fraction = removed / available if available else 1.0
    automatic["projected_leaf_excluded_fraction"] = fraction
    reason = None
    if touches_protected:
        reason = "whole-leaf-exclusion-touches-protected-support"
    elif fraction > document["automatic_avoidance"]["max_fraction"]:
        reason = "whole-leaf-area-budget-exceeded"
    elif (available - removed) / ((x1 - x0) * (y1 - y0)) < meter["min_area_fraction"]:
        reason = "whole-leaf-minimum-support"
    if reason:
        automatic["fallback"] = reason
        automatic["automatic_exclusions"] = []
        automatic["automatic_excluded_fraction"] = 0.0
        if "application_status" in automatic:
            automatic["application_status"] = "budget-fallback"


def _prepare(source, config, roi, document, source_key, highlight, *, checkpoint=None):
    identity = _identity(source, source_key)
    meter = _meter_config(config)
    render = _render_config(meter, document)
    _render_configuration(meter, render)
    effective_document = deepcopy(document)
    automatic = None
    if document.get("automatic_avoidance"):
        from .automatic_metering_v31 import detect_automatic_avoidance
        from .immutable_source_v44 import certificate_for

        certificate = certificate_for(source)
        from .film_density_domain_v45 import film_domain_for

        domain = film_domain_for(source)
        detection_source = domain.original if domain is not None else source
        def detect():
            return detect_automatic_avoidance(detection_source, roi, document["automatic_avoidance"])
        automatic = (certificate.automatic_result(roi, document["automatic_avoidance"], detect)
                     if certificate is not None else detect())
        if checkpoint is not None:
            checkpoint("automatic_detection")
        _project_automatic_support(source, roi, document, automatic)
        boxes = (
            effective_document["metering"]["excluded_rois"]
            + automatic["automatic_exclusions"]
        )
        _require(len(boxes) <= 64, "Automatic and manual exclusions exceed bound")
        effective_document["metering"]["excluded_rois"] = boxes
    reference = None
    if meter.process.process_mode == ProcessMode.BW:
        state = BWMaskedState(
            source, meter, roi, effective_document["metering"], identity=identity
        )
        if highlight:
            if highlight.get("automatic_reference"):
                from .highlight_workflow import resolve_spatial_highlight_reference

                restored = restore_zonal_metering(state.bundle.to_json(), identity)
                highlight = resolve_spatial_highlight_reference(highlight, restored)
            reference_roi = highlight["reference_roi"]
            _check_highlight_reference_bounds(reference_roi, source.shape)
            x0, y0, x1, y1 = reference_roi
            bx0, by0, bx1, by1 = state.roi
            reference_state = state
            if not (
                bx0 <= x0 < x1 <= bx1
                and by0 <= y0 < y1 <= by1
                and state._support[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0].all()
            ):
                refmeter = deepcopy(effective_document["metering"])
                refmeter.update(
                    include=[], exclude=[], excluded_rois=[], padding=0,
                    min_area_fraction=1.0, zones=[2, 2], blocks_per_zone=[2, 2],
                )
                reference_state = BWMaskedState(
                    source, meter, reference_roi, refmeter, identity=identity
                )
            reference_config = reference_state.register_reference_roi(reference_roi)
            record = reference_state._records[_digest(_config_document(reference_config))]
            reference = NativeSpatialCPU.__new__(NativeSpatialCPU)
            reference._initialize(
                reference_config,
                _render_config(reference_config, document),
                reference_config,
                identity,
                record["serialized"]["metrics"],
                "BW",
                json.loads(reference_state.to_json())["sha256"],
                reference_state.bundle.sha256,
            )
        base = NativeSpatialCPU.from_bw(state, meter, render)
        bundle = state.bundle
    else:
        packet, bundle = _c41_packet(source, meter, roi, effective_document, identity, checkpoint=checkpoint)
        base = NativeSpatialCPU.from_c41(packet, meter, render)
        if highlight:
            # Automatic references follow metering support. Explicit references
            # measure the independently chosen region of the complete source.
            restored = restore_zonal_metering(bundle.to_json(), identity)
            from .highlight_workflow import resolve_spatial_highlight_reference

            highlight = resolve_spatial_highlight_reference(highlight, restored)
            _check_highlight_reference_bounds(highlight["reference_roi"], source.shape)
            refdoc = deepcopy(effective_document)
            refdoc["metering"].update(
                include=[],
                exclude=[],
                excluded_rois=[],
                padding=0,
                min_area_fraction=1.0,
                zones=[2, 2],
                blocks_per_zone=[2, 2],
            )
            refpacket, _ = _c41_packet(
                source, meter, highlight["reference_roi"], refdoc, identity
            )
            reference = NativeSpatialCPU.from_c41(refpacket, meter, render)
    payload = dict(
        schema=SCHEMA,
        version=1,
        source_key=source_key,
        source_identity=identity,
        input_config=_config_document(config),
        roi=list(roi),
        metering_document=deepcopy(document["metering"]),
        automatic_document=deepcopy(document.get("automatic_avoidance")),
        automatic_result=automatic,
        effective_metering=effective_document["metering"],
        window=asdict(restore_zonal_metering(bundle.to_json(), identity).window),
        highlight_reference_roi=deepcopy(
            highlight["reference_roi"] if highlight else None
        ),
        base=base.to_json(),
        reference=reference.to_json() if reference else None,
        runtime=_runtime(),
    )
    payload["sha256"] = _digest(payload)
    return base, reference, payload


def _load_snapshot(snapshot, source, config, roi, document, source_key, highlight):
    _require(
        type(snapshot) is str and len(snapshot.encode()) <= MAX_BYTES,
        "Invalid spatial workflow snapshot",
    )

    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate spatial workflow snapshot key")
            result[key] = value
        return result

    data = json.loads(snapshot, object_pairs_hook=unique)
    _require(
        type(data) is dict
        and set(data)
        == {
            "schema",
            "version",
            "source_key",
            "source_identity",
            "input_config",
            "roi",
            "metering_document",
            "automatic_document",
            "automatic_result",
            "effective_metering",
            "window",
            "highlight_reference_roi",
            "base",
            "reference",
            "runtime",
            "sha256",
        },
        "Malformed spatial workflow snapshot fields",
    )
    _require(
        type(data) is dict
        and data.get("schema") == SCHEMA
        and type(data.get("version")) is int
        and data["version"] == 1,
        "Unknown spatial workflow snapshot",
    )
    _require(
        data.get("sha256") == _digest({k: v for k, v in data.items() if k != "sha256"}),
        "Spatial workflow snapshot checksum changed",
    )
    runtime = _runtime()
    # Existing frozen packets use the same metering and explicit-reference format.
    previous_runtimes = [
        {**runtime, "spatial_workflow_v31.py": digest}
        for digest in (
            "f6888eb0012f29858d477404da5d8d28daca7dcc58e3cd686640348d62b134bd",
            "965bd641dd471f5cff599536f7e899c8b862c35bf26d4bdd5fe1aad52cd8555c",
            "fa2cddb2e52f153c89153352640e7f9b5e9ef6a9e4653f6cdbd352de4f1892e7",
            "1ead28c792134e14e94635301daabf65bfb5e30f0805248e21340226ff68199f",
        )
    ]
    # These exact packets predate the runtime compatibility fix. All other
    # runtime components still have to match byte-for-byte.
    previous_runtimes.extend(
        {
            **runtime,
            "spatial_workflow_v31.py": spatial_hash,
            "native_spatial_v30.py": native_hash,
        }
        for spatial_hash, native_hash in (
            ("965bd641dd471f5cff599536f7e899c8b862c35bf26d4bdd5fe1aad52cd8555c", "de762007206390f2dcbd9723431f1f94df1cf9a1c7891ffd6ebd1c0d8f5615f4"),
            ("f6888eb0012f29858d477404da5d8d28daca7dcc58e3cd686640348d62b134bd", "0dddc5b5a00b0c39fd8c160afb4e17e1a54a8990d703b92140b9fff1eb9b7362"),
            ("fa2cddb2e52f153c89153352640e7f9b5e9ef6a9e4653f6cdbd352de4f1892e7", "0dddc5b5a00b0c39fd8c160afb4e17e1a54a8990d703b92140b9fff1eb9b7362"),
            ("acb13c78fe25a423e49a44ee213123640638e11edd86cf4371a3aa25230a31e4", "0dddc5b5a00b0c39fd8c160afb4e17e1a54a8990d703b92140b9fff1eb9b7362"),
        )
    )
    _require(
        data["runtime"] in [runtime, *previous_runtimes]
        and data["source_key"] == source_key
        and data["input_config"] == _config_document(config)
        and data["roi"] == list(roi)
        and data["metering_document"] == document["metering"]
        and data["automatic_document"] == document.get("automatic_avoidance"),
        "Spatial workflow source/config/metering changed",
    )
    _check_source(source, data["source_identity"])
    requested_ref = highlight["reference_roi"] if highlight else None
    if highlight and highlight.get("automatic_reference"):
        from .highlight_workflow import highlight_support, select_highlight_reference

        recipe = dict(roi=list(roi), native_spatial_controls={"metering": data["effective_metering"]})
        requested_ref = select_highlight_reference(recipe, highlight_support(recipe, source.shape))
    # OFF may reuse an already frozen reference; a new reference requires a new job.
    _require(
        requested_ref is None or requested_ref == data["highlight_reference_roi"],
        "Changed highlight reference requires a new spatial job",
    )
    runners = []
    for text in (data["base"], data["reference"] if requested_ref else None):
        if text is None:
            runners.append(None)
            continue
        packet = json.loads(text)
        meter = WorkspaceConfig.from_flat_dict(deepcopy(packet["meter_config"]))
        old_render = WorkspaceConfig.from_flat_dict(deepcopy(packet["render_config"]))
        restored = NativeSpatialCPU.restore(text, source, meter, old_render)
        new_render = _render_config(meter, document)
        # Change render strengths only; the restored auxiliary packet is retained.
        runner = NativeSpatialCPU.__new__(NativeSpatialCPU)
        runner._initialize(
            meter,
            new_render,
            restored._effective_meter,
            restored._identity,
            restored._payload["metrics"],
            restored._payload["parent_type"],
            restored._payload["parent_binding_sha256"],
            restored._payload["density_bundle_sha256"],
        )
        runners.append(runner)
    return *runners, data


def validate_spatial_replay(
    snapshot, source, config, roi, document, source_key, highlight, film=None
):
    """Validate replay; film calibration also authenticates its derived log."""
    if film and film["strength"] != 0:
        from .film_density_domain_v45 import validate_film_spatial

        return validate_film_spatial(snapshot, source, config, roi, document, source_key, highlight, film)
    readonly = source if not source.flags.writeable else source.view()
    readonly.setflags(write=False)
    _load_snapshot(
        snapshot,
        readonly,
        config,
        roi,
        checked_spatial_controls(document),
        source_key,
        highlight,
    )


def _highlight_mask_blocks(
    source_shape, source_key, document, block_rows=64, *, tile_pixel_limit=4_194_304
):
    """Yield exact full-source masks in bounded row blocks for one-shot exports."""
    height, width = source_shape[:2]
    if not document:
        for _ in range(0, height, block_rows):
            yield None
        return

    # Extremely wide images exceed the sampled-mask pixel limit even for one
    # rendering block. Keep the prior full-frame implementation for them.
    limit = tile_pixel_limit
    if width * block_rows > limit:
        from .highlight_protection import build_highlight_protection

        mask = build_highlight_protection(
            source_shape, source_key, dtype="float64", **document
        ).mask
        for first in range(0, height, block_rows):
            yield mask[first:first + block_rows]
        return

    from .highlight_sample_v46 import sample_highlight_protection

    tile_rows = max(block_rows, (limit // width // block_rows) * block_rows)
    for tile_start in range(0, height, tile_rows):
        tile_end = min(height, tile_start + tile_rows)
        tile = sample_highlight_protection(
            list(source_shape), source_key, **document,
            rectangle=[0, tile_start, width, tile_end],
            sample_step=1, dtype="float64",
        ).mask
        for offset in range(0, len(tile), block_rows):
            yield tile[offset:offset + block_rows]


def encode_spatial_workflow(
    processor,
    source,
    config,
    records,
    roi,
    document,
    *,
    source_key,
    snapshot=None,
    highlight=None,
    point=None,
    film=None,
):
    """Original full-source CPU -> optional guarded high/point -> SDK RGB16/ICC."""
    if film and film["strength"] != 0:
        from .film_density_domain_v45 import encode_film_spatial

        return encode_film_spatial(processor, source, config, records, roi, document,
                                  film=film, source_key=source_key, snapshot=snapshot,
                                  highlight=highlight, point=point)
    from .highlight_reference_guarded import apply_highlight_reference_guarded
    from .highlight_workflow import _merge_highlight
    from .native_float import checked_native_float
    from .positive_float import (
        adapt_positive_float,
        apply_positive_float,
        positive_domain,
        working_icc_bytes,
    )
    from negpy.kernel.image.logic import working_oetf_encode

    document = checked_spatial_controls(document)
    _require(document is not None, "Active spatial export requires active controls")
    _require(
        isinstance(source, np.ndarray)
        and source.dtype == np.float32
        and source.ndim == 3
        and source.shape[-1] == 3
        and source.flags.c_contiguous,
        "Spatial export requires original C-order F32 HWC3",
    )
    original_sha = _pixels(source)
    readonly = source if not source.flags.writeable else source.view()
    readonly.setflags(write=False)
    if snapshot is None:
        base, reference, frozen = _prepare(
            readonly, config, roi, document, source_key, highlight
        )
    else:
        base, reference, frozen = _load_snapshot(
            snapshot, readonly, config, roi, document, source_key, highlight
        )
    # The raw records are part of the caller source key, never used as surrogate image data.
    _require(
        bool(records) and all(type(x.get("sha256")) is str for x in records),
        "Explicit source records required",
    )
    # Both renderers are local to this one-shot export. Retaining their full
    # intermediate stages cannot accelerate another render and multiplies the
    # peak memory needed by large scans.
    base_result = base.render(readonly, base._meter, base._render, cache_stages=False)
    image = base_result.image
    ref_image = (
        reference.render(readonly, reference._meter, reference._render, cache_stages=False).image
        if reference
        else None
    )
    point = checked_native_float(point)
    high_receipt = None
    if highlight or point["values"]:
        icc = working_icc_bytes()
        encoded_domain = positive_domain("working-oetf", icc)
        linear_domain = positive_domain("linear", icc)
        protection_masks = _highlight_mask_blocks(
            source.shape, source_key, highlight["protection"] if highlight else None
        )
        effect_masks = _highlight_mask_blocks(
            source.shape, source_key, highlight.get("effect_area") if highlight else None
        )
        final_image = np.empty(source.shape, dtype=np.float32)
        for start in range(0, len(image), 64):
            part = image[start : start + 64]
            linear = adapt_positive_float(
                part, domain=encoded_domain, icc_bytes=icc
            ).image
            output = linear
            if highlight:
                ref = adapt_positive_float(
                    ref_image[start : start + 64], domain=encoded_domain, icc_bytes=icc
                ).image
                output, observation = apply_highlight_reference_guarded(
                    linear,
                    ref,
                    highlight["controls"],
                    domain=linear_domain,
                    protection_mask=next(protection_masks),
                    effect_mask=next(effect_masks),
                )
                high_receipt = _merge_highlight(high_receipt, observation)
            if point["values"]:
                output = apply_positive_float(
                    output, point, domain=linear_domain, icc_bytes=icc
                ).image
            encoded = working_oetf_encode(np.clip(output, 0, 1))
            if not point["values"]:
                unchanged = np.all(output == linear, axis=2)
                encoded[unchanged] = part[unchanged]
            final_image[start : start + len(part)] = encoded
        image = final_image
    export = replace(
        config.export,
        export_fmt=ExportFormat.TIFF,
        export_bit_depth=16,
        export_color_space=ColorSpace.ADOBE_RGB.value,
        tiff_compression=TiffCompression.ZIP,
    )
    payload, status = processor.encode_export(image, export, WORKING_COLOR_SPACE)
    if payload is None:
        raise RuntimeError(status)
    _require(_pixels(source) == original_sha, "Spatial export source changed")
    _check_source(readonly, frozen["source_identity"])
    report = dict(
        schema=SCHEMA,
        source_pixels_sha256=original_sha,
        input_config_sha256=_digest(_config_document(config)),
        spatial_controls=deepcopy(document),
        metering_window=deepcopy(frozen["window"]),
        automatic_avoidance=frozen["automatic_result"],
        **({"highlight_reference_roi": frozen["highlight_reference_roi"]}
           if highlight and highlight.get("automatic_reference") else {}),
        high_processing=high_receipt,
        source_values_changed=False,
        full_source_processing=True,
        C41_reference_policy="separate clean zonal reference"
        if highlight and config.process.process_mode == ProcessMode.C41
        else None,
        snapshot_restored_without_metering=snapshot is not None,
        output_encoded_sha256=_pixels(image),
    )
    return payload, export, report, _canonical(frozen)
