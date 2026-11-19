"""One strict, measurement-only masked density context for color renderers.

The existing spatial meter and unchanged Native encoder produce an independent
neutral reference. Creative Native Lab/local/point controls never cross this
boundary; Nikon consumes that reference and Flex consumes the same window.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np


def checked_density_metering(document):
    from .bw_metering import checked_bw_metering
    from .automatic_metering_v31 import checked_automatic_avoidance

    if (type(document) is not dict or not {"version", "metering"} <= set(document)
            or set(document) - {"version", "metering", "automatic_avoidance"}
            or type(document["version"]) is not int or document["version"] != 1):
        raise ValueError("共享密度测量需要严格的版本化测量文档")
    try:
        if len(json.dumps(document, allow_nan=False).encode()) > 512 * 1024:
            raise ValueError("共享密度测量文档过大")
    except (TypeError, OverflowError) as error:
        raise ValueError("共享密度测量必须为有限JSON参数") from error
    meter = checked_bw_metering(document["metering"])
    if meter is None:
        raise ValueError("共享密度测量需要活动测量区域")
    result = dict(version=1, metering=meter)
    if "automatic_avoidance" in document:
        automatic = checked_automatic_avoidance(document["automatic_avoidance"])
        if automatic is not None:
            result["automatic_avoidance"] = automatic
    return result


def spatial_document(document):
    document = checked_density_metering(document)
    return dict(version=1, enabled=True, lab={}, local_masks=[],
                **{key: deepcopy(value) for key, value in document.items() if key != "version"})


def window_from_document(document):
    from .density_matrix import DensityWindow
    from .engines import Analysis

    if (type(document) is not dict
            or set(document) != {"provider", "analysis", "log_dense", "log_thin", "metadata"}
            or type(document["provider"]) is not str or not document["provider"]
            or type(document["metadata"]) is not dict):
        raise ValueError("共享密度测量缺少完整冻结窗口")
    value = document["analysis"]
    if type(value) is not dict or set(value) != {"dark", "base", "roi", "percentiles"}:
        raise ValueError("共享密度测量缺少完整Analysis")
    roi = value["roi"]
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(type(v) is not int for v in roi)
            or not 0 <= roi[0] < roi[2] or not 0 <= roi[1] < roi[3]):
        raise ValueError("共享密度窗口的源区域无效")
    arrays = []
    for values in (value["dark"], value["base"], document["log_dense"], document["log_thin"]):
        if (not isinstance(values, (list, tuple)) or len(values) != 3
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)):
            raise ValueError("共享密度窗口需要有限RGB三通道端点")
        arrays.append(np.asarray(values, dtype=np.float64))
    dark, base, dense, thin = arrays
    if (np.any(dark <= 0) or np.any(base <= dark) or np.any(thin <= dense)
            or not np.allclose(np.log10(dark), dense, rtol=0, atol=1e-12)
            or not np.allclose(np.log10(base), thin, rtol=0, atol=1e-12)):
        raise ValueError("共享密度窗口的信号与对数端点不一致")
    percentiles = value["percentiles"]
    if percentiles is not None and (
        not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2
        or any(type(v) not in (int, float) or not np.isfinite(v) for v in percentiles)
        or not 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        raise ValueError("共享密度窗口分位数无效")
    analysis = Analysis(tuple(dark), tuple(base), tuple(roi),
                        tuple(percentiles) if percentiles is not None else None)
    return DensityWindow(document["provider"], analysis, tuple(dense), tuple(thin),
                         deepcopy(document["metadata"]))


def render_shared_measurement(processor, source, config, records, roi, document,
                              *, source_key, snapshot=None):
    from .spatial_workflow_v31 import encode_spatial_workflow

    checked = checked_density_metering(document)
    payload, export, info, frozen = encode_spatial_workflow(
        processor, source, config, records, roi, spatial_document(checked),
        source_key=source_key, snapshot=snapshot,
    )
    window = window_from_document(info["metering_window"])
    if list(window.analysis.roi) != list(roi):
        raise ValueError("共享密度窗口与当前源区域不一致")
    info = {**info, "measurement_only": True, "native_reference": True,
            "contract_document": checked}
    return payload, export, info, frozen, window


def validate_saved_shared_measurement(manifest):
    """Validate the frozen measurement proof before using saved pixels/windows.

    This is a metadata check, not a replacement for source authentication and
    the spatial producer's full snapshot replay against original source data.
    """
    from .spatial_workflow_v31 import SCHEMA, MAX_BYTES, _runtime

    document = checked_density_metering(manifest["recipe"]["density_metering"])
    info = manifest.get("shared_measurement_processing")
    text = manifest.get("shared_measurement_snapshot")
    if (type(info) is not dict or type(text) is not str or not text
            or len(text.encode()) > MAX_BYTES
            or info.get("measurement_only") is not True or info.get("native_reference") is not True
            or info.get("contract_document") != document
            or info.get("spatial_controls") != spatial_document(document)
            or info.get("metering_window") != manifest.get("density_window")):
        raise ValueError("共享密度保存结果缺少完整测量证明")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("共享密度快照含重复字段")
            result[key] = value
        return result

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    try:
        data = json.loads(text, object_pairs_hook=unique)
        if (type(data) is not dict or set(data) != {
            "schema", "version", "source_key", "source_identity", "input_config", "roi",
            "metering_document", "automatic_document", "automatic_result", "effective_metering",
            "window", "highlight_reference_roi", "base", "reference", "runtime", "sha256",
        } or data["schema"] != SCHEMA or type(data["version"]) is not int or data["version"] != 1
                or data["sha256"] != digest({key: value for key, value in data.items() if key != "sha256"})
                or data["runtime"] != _runtime()
                or data["input_config"] != manifest["native_config"]
                or data["roi"] != manifest["roi"]
                or data["metering_document"] != document["metering"]
                or data["automatic_document"] != document.get("automatic_avoidance")
                or data["automatic_result"] != info.get("automatic_avoidance")
                or data["window"] != info["metering_window"]
                or data["highlight_reference_roi"] is not None or data["reference"] is not None
                or data["source_identity"]["shape"] != manifest["source_shape"]
                or data["source_identity"]["source_pixels_sha256"] != info.get("source_pixels_sha256")
                or info.get("input_config_sha256") != digest(manifest["native_config"])):
            raise ValueError("共享密度快照与源网格、测量支持或中性配置不一致")
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError("共享密度冻结快照无效") from error
    window = window_from_document(info["metering_window"])
    if list(window.analysis.roi) != manifest["roi"]:
        raise ValueError("共享密度窗口与保存源区域不一致")
    return window
