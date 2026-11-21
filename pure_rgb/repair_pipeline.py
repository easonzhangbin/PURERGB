"""Apply a confirmed dust plan after colour, without changing render anchors."""

from pathlib import Path

import numpy as np


def render_repair(request, manifest, source_record, mode, *, directory, on_progress):
    """Keep the colour baseline immutable and publish one repaired derivative."""
    plan = request.get("repair_plan")
    if plan is None:
        return None
    from .repair_plan import validate_frame
    from .repair_replay import replay_record

    validate_frame(plan, manifest["recipe"], manifest, mode)
    record = replay_record(source_record, plan, directory=directory, on_progress=on_progress)
    manifest.setdefault("repair_results", {})[mode] = record
    return record


def preview_repair(request, session, pixels, reference, icc, *, on_progress):
    """Repair final display grids once, after all session rendering branches.

    Sessions always retain unretouched colour anchors. The displayed result and
    comparison are derived afresh, so a previous repair cannot be applied twice.
    """
    plan = request.get("repair_plan")
    if plan is None:
        return pixels, reference, None
    from .repair_plan import validate_frame
    from .repair_replay import replay_pixels

    recipe = request.get("draft_recipe")
    if recipe is None:
        recipe = {**session.recipe, **request.get("controls", {})}
    validate_frame(plan, recipe, session.manifest, request["mode"])
    same = pixels is reference or np.array_equal(pixels, reference)
    viewport = getattr(session, "viewport_result", None)
    options = dict(
        directory=Path(request["directory"]),
        output_size=getattr(session, "output_size", None),
        sample_step=session.sample_step,
        viewport=viewport,
        on_progress=on_progress,
    )
    current, provenance = replay_pixels(pixels, icc, plan, **options)
    if same:
        compared = current
    else:
        compared, _ = replay_pixels(reference, icc, plan, **options)
    return current, compared, provenance


def export_record(manifest, mode, *, repair_required=False):
    """Select a verified derivative only when bound to this colour baseline."""
    source = manifest["results"][mode]
    record = manifest.get("repair_results", {}).get(mode)
    if record is None:
        if repair_required:
            raise ValueError("已确认除尘尚未应用到新成片，导出未继续")
        return source
    if (record.get("source_sha256") != source["sha256"]
            or record.get("outside_mask_changed_pixels") != 0):
        raise ValueError("除尘派生成片与当前输出不一致")
    from .repair_plan import validate_frame

    validate_frame(record["repair_plan"], manifest["recipe"], manifest, mode)
    return record
