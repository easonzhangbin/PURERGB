"""Read-only definitions for source-domain metering overlays and real receipts."""

from bisect import bisect_right
from copy import deepcopy


_FALLBACKS = {
    "advisory-only": "当前策略仅提供建议",
    "area-budget-exceeded": "候选面积超过自动排除上限",
    "contour-box-budget-exceeded": "候选区域数量超过自动排除上限",
    "whole-leaf-area-budget-exceeded": "实际分区排除面积超过上限",
    "whole-leaf-minimum-support": "排除后剩余测量支持不足",
    "whole-leaf-exclusion-touches-protected-support": "排除会触及受保护的测量分区",
}


def _rectangle(bounds):
    return dict(type="rectangle", bounds=list(bounds))


def _overlaps(first, second):
    a, b, c, d = first
    x, y, z, w = second
    return a < z and x < c and b < w and y < d


def _candidate_reasons(report, box):
    reasons = []
    for candidate in report.get("candidates", []):
        if not any(_overlaps(box, bound) for bound in candidate.get("boxes", [])):
            continue
        kind = candidate.get("kind", "")
        reason = {
            "edge-connected-clipped-plateau": "边缘连通的截断平台",
            "edge-signal-pattern": "边缘信号模式",
        }.get(kind)
        if kind.startswith("common-density-"):
            reason = "边缘公共密度偏离"
        elif kind.startswith("channel-"):
            reason = "边缘通道密度差异"
        if reason and reason not in reasons:
            reasons.append(reason)
        context = candidate.get("source_context_check", {})
        if context.get("eligible"):
            text = "通过源画面上下文核对"
            if text not in reasons:
                reasons.append(text)
    return "、".join(reasons) or "回执未提供该区域的检测依据"


def build_metering_hover(recipe, receipt, shown_recipe, roi, protected_rois=(), *,
                         show_grid=True, review_grid=False):
    """Describe settings without measuring, changing them or trusting outline color.

    An applied label needs both the matching density receipt and the exact
    recipe currently shown. A report may describe an older result; those
    boundaries remain explicitly pending rather than being called applied.
    """
    spatial = recipe.get("native_spatial_controls") or recipe.get("density_metering") or {}
    meter = spatial.get("metering") or {}
    automatic = spatial.get("automatic_avoidance")
    receipt = receipt or {}
    matched = bool(receipt and receipt.get("roi") == list(roi)
                   and receipt.get("metering") == spatial.get("metering")
                   and receipt.get("automatic") == automatic)
    fresh = bool(matched and shown_recipe == recipe)
    status = "已生效" if fresh else "待更新"
    regions = []

    def add(kind, shape, title, *details, applied=None):
        regions.append(dict(kind=kind, shape=deepcopy(shape), title=title,
                            status=status if applied is None else applied,
                            details=list(details)))

    add("roi", _rectangle(roi), "测量范围",
        "在此范围内统计密度；输出裁切由画幅设置决定。")
    for shape in meter.get("include", []):
        add("include", shape, "手工测量选区",
            "仅使用选区与测量范围的交集；多个选区取并集。")
    for shape in meter.get("exclude", []):
        add("manual_exclusion", shape, "手工排除",
            "从密度统计中移除此形状；源像素不变。")
    padding = meter.get("padding", 0)
    for box in meter.get("excluded_rois", []):
        a, b, c, d = box
        add("manual_exclusion", _rectangle([a - padding, b - padding, c + padding, d + padding]),
            "手工分区排除", "与此范围相交的测量小分区不参与统计。")
    protected = list(protected_rois)
    if automatic:
        for box in automatic.get("protected_rois", []):
            if box not in protected:
                protected.append(box)
    for box in protected:
        if automatic and automatic.get("enabled"):
            protected_status = status if box in automatic.get("protected_rois", []) else "待更新"
        else:
            protected_status = "未启用"
        add("protected", _rectangle(box), "自动避让保护区",
            "阻止自动排除触及此处；手工排除仍优先。",
            applied=protected_status)

    # Never project a different support's report onto the current settings.
    report = receipt.get("report", {}) if matched else {}
    action = (automatic or {}).get("action", report.get("application_action", "exclude"))
    fallback = report.get("fallback")
    legacy_exclude = bool(automatic and automatic.get("version") == 1
                          and report.get("version") == 1
                          and report.get("document") == automatic
                          and automatic.get("mode") != "advisory"
                          and "application_action" not in report)
    can_apply = bool(fresh and automatic and automatic.get("enabled")
                     and action == "exclude" and not fallback
                     and (report.get("application_action") == "exclude" or legacy_exclude))
    applied_boxes = report.get("automatic_exclusions", [])
    candidates = list(report.get("candidate_exclusions", []))
    for box in applied_boxes:
        if box not in candidates:
            candidates.append(box)
    for box in candidates:
        applied = can_apply and box in applied_boxes
        details = ["检测依据：" + _candidate_reasons(report, box)]
        if fallback:
            details.append("未自动排除：" + _FALLBACKS.get(fallback, "检测回执要求保留测量支持"))
        elif action == "suggest":
            details.append("当前仅建议，此区域仍参与测量。")
        elif applied:
            details.append("回执确认已从密度统计中自动排除相交小分区。")
        else:
            details.append("尚未确认在当前显示结果中自动排除。")
        details.append("候选不代表已判定底片缺陷。")
        add("auto_exclusion" if applied else "auto_candidate", _rectangle(box),
            "自动排除" if applied else "自动避让候选", *details,
            applied="已生效" if applied else "待更新" if not fresh else "仅建议" if action == "suggest" else "未生效")
    grid = None
    if show_grid:
        records = receipt.get("zones") or []
        grid = dict(roi=list(roi), zones=list(meter.get("zones", [8, 12])), fresh=fresh)
        if fresh and records:
            grid["records"] = deepcopy(records)
        if review_grid:
            grid["review"] = True
            grid["excluded_rectangles"] = [
                list(shape["bounds"]) for shape in meter.get("exclude", [])
                if shape["type"] == "rectangle"
            ]
    return dict(regions=regions, grid=grid, fresh=fresh)


def grid_hover(grid, point):
    """Return the original integer metering row/column, never rounded fractions."""
    if not grid:
        return None
    x, y = point
    x0, y0, x1, y1 = grid["roi"]
    if not (x0 <= x < x1 and y0 <= y < y1):
        return None
    rows, columns = grid["zones"]
    xs = [x0 + (x1 - x0) * col // columns for col in range(columns + 1)]
    ys = [y0 + (y1 - y0) * row // rows for row in range(rows + 1)]
    row, col = bisect_right(ys, y), bisect_right(xs, x)
    records = grid.get("records", [])
    index = (row - 1) * columns + col - 1
    details = ["按当前有效选区统计；排除部分不参与密度测量。"]
    status = "已生效" if grid["fresh"] else "待更新"
    if index < len(records):
        stats = records[index]
        details = [f"有效像素 {stats['supported_pixel_area']:,} / {stats['total_pixel_area']:,}"
                   f"（{stats['retained_fraction']:.1%}）；有效子区 "
                   f"{stats['valid_leaf_count']} / {stats['total_leaf_count']}"]
        median = stats["median_log_rgb"]
        if median is None:
            status = "无有效样本"
        else:
            details.append("RGB 光密度 D＝−log₁₀(T) · 中位 "
                           + " / ".join(f"{-value:.3f}" for value in median))
            low, high = stats["low_log_rgb"], stats["high_log_rgb"]
            details.append("分位范围 · " + " / ".join(
                f"{-upper:.3f}–{-lower:.3f}" for lower, upper in zip(low, high)
            ))
    result = dict(kind="grid", title=f"测量分区 · 第 {row} 行 / 第 {col} 列",
                  status=status, details=details, row=row, column=col)
    if grid.get("review"):
        result["bounds"] = [xs[col - 1], ys[row - 1], xs[col], ys[row]]
        if index < len(records):
            result["supported_pixels"] = records[index]["supported_pixel_area"]
    return result


def metering_zone_records(window):
    """Read the exact per-zone statistics already produced by the meter."""
    return (window.get("metadata", {}).get("zonal_metering") or {}).get("zone_records", [])


def hover_text(region):
    return "\n".join([region["title"] + " · " + region["status"], *region["details"]])
