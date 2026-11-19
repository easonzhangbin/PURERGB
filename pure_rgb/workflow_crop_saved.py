"""Prove that an existing positive can supply the complete crop editor.

Only metadata is read here. Pixel authentication and display conversion remain
in the existing display worker, away from the UI thread. A stale color version
or already-cropped/warped TIFF is never substituted for the current image.
"""

from copy import deepcopy
import json
from pathlib import Path

from .output_orientation import checked_document
from .output_orientation_cache import _color_recipe
from .workspace import checked_recipe, output_rectangle


_MODES = {"native": "native", "nikon": "cool-shared-rgb", "flex": "flex-endpoint"}


def _comparable(recipe, mode):
    value = deepcopy(_color_recipe(recipe, mode))
    value.pop("crop", None)
    value.pop("output_roi", None)
    return value


def _resolved_recipe(recipe, auto_roi):
    result = checked_recipe(recipe, verify_resources=False)
    if result.get("roi") is None and auto_roi is not None:
        # Historical initial saves retain roi=None in their recipe while the
        # completed receipt records the resolved analysis window. It is the
        # same measured support as the normally resolved UI draft.
        result = checked_recipe({**result, "roi": deepcopy(auto_roi)}, verify_resources=False)
    return result


def saved_full_source_record(window, locator_recipe, *, allow_reference=False):
    """Return an exactly matching full-source positive, with its saved D4.

    The caller aligns the authenticated image from this saved orientation to
    the locator recipe's current direction. The recipe itself is not changed.
    An unusable candidate is skipped so the caller can render a fresh locator.
    """
    photo = window.current_photo()
    if not photo or not photo.get("source_shape"):
        return None
    mode = _MODES[photo["engine"]]
    shape = list(photo["source_shape"])
    try:
        desired = _resolved_recipe(locator_recipe, photo.get("analysis_roi"))
    except (ValueError, TypeError, KeyError, OSError):
        return None
    desired_recipe = desired
    desired = _comparable(desired, mode)
    ids = [photo.get("saved", {}).get(photo["engine"]),
           *reversed(photo.get("history", [])), photo.get("initial_job")]
    visited = set()
    for job_id in ids:
        if not job_id or job_id in visited:
            continue
        visited.add(job_id)
        try:
            path = window.project.workspace.root / "runs" / job_id / "manifest.json"
            manifest = json.loads(path.read_text())
            if manifest.get("state") != "complete" or manifest.get("source_shape") != shape:
                continue
            saved = _resolved_recipe(manifest["recipe"], manifest.get("roi"))
            candidate_mode = mode
            if allow_reference and candidate_mode not in manifest["results"]:
                candidate_mode = "native"
            if allow_reference and saved["raw"] != desired_recipe["raw"]:
                continue
            if not allow_reference and _comparable(saved, mode) != desired:
                continue
            values = saved.get("post_controls", {}).get(candidate_mode, {}).get("values", {})
            if any(key.startswith("geometry.") for key in values):
                continue
            if output_rectangle(saved, manifest["roi"], shape) != [0, 0, shape[1], shape[0]]:
                continue
            direction = checked_document(saved.get("output_orientation"))
            expected = list(shape)
            if direction and direction["quarter_turns"] % 2:
                expected[0], expected[1] = expected[1], expected[0]
            record = manifest["results"].get(candidate_mode)
            if (not record or record.get("shape") != expected
                    or not Path(record["path"]).is_file()):
                continue
            from .draft_preview import _saved_measurement_valid

            if not _saved_measurement_valid(manifest):
                continue
            record = deepcopy(record)
            record["display_source_step"] = max(1, (max(expected[:2]) + 2047) // 2048)
            return dict(record=record, output_orientation=deepcopy(direction),
                        manifest=str(path), mode=candidate_mode, reference=allow_reference)
        except (ValueError, TypeError, KeyError, OSError, IndexError):
            continue
    return None
