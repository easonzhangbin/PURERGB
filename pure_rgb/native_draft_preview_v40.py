"""Use the already frozen automatic ROI when comparing preview recipes.

An original saved recipe can keep ``roi=None`` to record that automatic
detection was requested.  The completed manifest separately records the exact
rectangle which was used.  UI drafts make that rectangle explicit; comparing
the request marker to its resolved value must not invalidate a color anchor.
"""

from copy import deepcopy

from .workspace import checked_recipe


def resolved_saved_recipe(manifest):
    """Return an effective recipe without editing a saved manifest or its ROI.

    Only the automatic-request marker is resolved. An explicitly different
    ROI remains different, so changing the actual analysis support still
    invalidates frozen-analysis and saved-positive reuse.
    """
    recipe = checked_recipe(manifest["recipe"])
    if recipe.get("roi") is not None:
        return recipe
    roi, shape = manifest.get("roi"), manifest.get("source_shape")
    if (
        not isinstance(roi, (list, tuple))
        or len(roi) != 4
        or any(type(value) is not int for value in roi)
        or not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or shape[2] != 3
    ):
        raise ValueError("已保存的自动测量区域缺少完整源网格坐标")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("已保存的自动测量区域超出完整源网格")
    result = deepcopy(recipe)
    result["roi"] = list(roi)
    return checked_recipe(result)
