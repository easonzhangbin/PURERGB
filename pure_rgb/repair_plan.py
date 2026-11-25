"""Bind confirmed dust selections to a source and retain their review coordinates.

Colour and supported output geometry may change. Source pixels and the review
evidence may not. Large TIFF/mask authentication belongs to the repair
executor; this module authenticates the small provenance documents.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .workspace import output_rectangle, source_recipe_identity


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("除尘记录路径无效")
    return str(Path(value).expanduser().resolve())


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("除尘记录缺少有效 SHA256")
    return value


def _inputs(records):
    if not isinstance(records, list) or not records:
        raise ValueError("除尘记录缺少来源文件")
    return [dict(role=r["role"], path=_path(r["path"]), sha256=_sha(r["sha256"])) for r in records]


def _calibration(value):
    if not value:
        return None
    return {key: value[key] for key in ("path", "sha256", "gain_sha256") if key in value}


def _raw_identity(manifest):
    receipt = manifest.get("raw_import_receipt") or {}
    return {key: receipt[key] for key in ("source_pixels_sha256", "geometry_token") if receipt.get(key)}


def same_optics_baseline_recipe(original, recipe):
    identities = [source_recipe_identity(value) for value in (original, recipe)]
    for identity in identities:
        identity.pop("chromatic_aberration", None)
        identity.pop("lens_profile", None)
    return identities[0] == identities[1]


def _same_chromatic_source(parent, recipe, manifest):
    """Allow CA-only R/B resampling when its unchanged baseline is evidenced.

    Green is the coordinate reference for dust coverage. The full pre-CA RGB
    hash must also match, so this cannot authorize new RAWs, flat fields,
    registration, distortion, or merely equal-sized unrelated pictures.
    """
    if not same_optics_baseline_recipe(parent["recipe"], recipe):
        return False
    baselines, references = [], 0
    for document in (parent, manifest):
        reference = (document.get("raw_import_receipt") or {}).get("chromatic_reference")
        if reference is not None:
            if (reference.get("green_unchanged") is not True
                    or reference.get("canvas_unchanged") is not True):
                return False
            _sha(reference.get("green_pixels_sha256"))
            baselines.append({key: _sha(reference.get(key))
                              for key in ("source_pixels_sha256", "geometry_token")})
            references += 1
        else:
            baseline = _raw_identity(document)
            if set(baseline) != {"source_pixels_sha256", "geometry_token"}:
                return False
            baselines.append(baseline)
    return bool(references and baselines[0] == baselines[1])


def geometry_key(recipe, manifest, mode):
    """Identity of source-to-output coordinates, independent of colour values."""
    shape = list(manifest["source_shape"])
    analysis = recipe.get("roi") or manifest["roi"]
    values = recipe.get("post_controls", {}).get(mode, {}).get("values", {})
    return _digest(dict(mode=mode, source=source_recipe_identity(recipe), source_shape=shape,
                        rectangle=output_rectangle(recipe, analysis, shape),
                        orientation=recipe.get("output_orientation"),
                        geometry={k: v for k, v in values.items() if k.startswith("geometry.")}))


def _coordinate_key(recipe, manifest, mode):
    """Compare rendered coordinates across engines without changing legacy hashes."""
    shape = list(manifest["source_shape"])
    analysis = recipe.get("roi") or manifest["roi"]
    values = recipe.get("post_controls", {}).get(mode, {}).get("values", {})
    return _digest(dict(source=source_recipe_identity(recipe), source_shape=shape,
                        rectangle=output_rectangle(recipe, analysis, shape),
                        orientation=recipe.get("output_orientation"),
                        geometry={k: v for k, v in values.items() if k.startswith("geometry.")}))


def _document(binding):
    path = Path(_path(binding["path"]))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(binding["sha256"]):
        raise ValueError("除尘审核或来源清单已改变，请重新审核")
    return json.loads(raw)


def checked_plan(plan):
    """Authenticate descriptor and JSON evidence without reading full rasters."""
    try:
        if not isinstance(plan, dict) or type(plan.get("version")) is not int or plan["version"] != 1:
            raise ValueError("不支持的除尘继承记录")
        result = deepcopy(plan)
        shape = result["shape"]
        source_shape = result["binding"]["source_shape"]
        for size in (shape, source_shape):
            if (not isinstance(size, list) or len(size) != 3 or size[-1] != 3
                    or any(type(n) is not int or n <= 0 for n in size)):
                raise ValueError("除尘审核画幅无效")
        _sha(result["mask_sha256"])
        _path(result["mask_path"])
        _inputs(result["binding"]["inputs"])
        _sha(result["binding"]["geometry_key"])
        _sha(result["backend"]["fingerprint"])
        _path(result["backend"]["root"])
        receipt = _document(dict(path=result["receipt_path"], sha256=result["receipt_sha256"]))
        parent = _document(result["parent_manifest"])
        mode = result["binding"]["mode"]
        source = receipt["source"]
        original = parent["results"][mode]
        preview = receipt["preview"]
        if (parent.get("state") != "complete" or receipt.get("state") != "confirmed" or receipt.get("revision") != source["sha256"]
                or source["sha256"] != original["sha256"]
                or _path(source["path"]) != _path(original["path"])
                or preview.get("shape") != shape
                or _path(preview["mask_path"]) != _path(result["mask_path"])
                or preview["mask_sha256"] != result["mask_sha256"]
                or receipt["backend"] != result["backend"]
                or _inputs(parent["inputs"]) != _inputs(result["binding"]["inputs"])
                or _calibration(parent.get("calibration")) != _calibration(result["binding"].get("calibration"))
                or parent["source_shape"] != source_shape
                or _raw_identity(parent) != result["binding"].get("raw_import_identity", {})
                or geometry_key(parent["recipe"], parent, mode) != result["binding"]["geometry_key"]):
            raise ValueError("除尘审核与原始成片的来源或画幅不一致")
        return result
    except (KeyError, TypeError, IndexError, OSError, json.JSONDecodeError) as error:
        raise ValueError("除尘审核证据不完整，请重新审核") from error


def validate_frame(plan, recipe, manifest, mode):
    """Return a mapping for edited output geometry, retaining authenticated masks."""
    plan = checked_plan(plan)
    binding = plan["binding"]
    try:
        parent = _document(plan["parent_manifest"])
        matching_source = (
            source_recipe_identity(recipe) == source_recipe_identity(parent["recipe"])
            and all(_raw_identity(manifest).get(key) == value
                    for key, value in binding.get("raw_import_identity", {}).items())
        ) or _same_chromatic_source(parent, recipe, manifest)
        if mode == binding["mode"]:
            matching_coordinates = geometry_key(recipe, manifest, mode) == binding["geometry_key"]
        else:
            matching_coordinates = (
                _coordinate_key(recipe, manifest, mode)
                == _coordinate_key(parent["recipe"], parent, binding["mode"])
            )
        if (_inputs(manifest["inputs"]) != _inputs(binding["inputs"])
                or [_path(p) for p in recipe["raw"]] != [r["path"] for r in _inputs(binding["inputs"])]
                or manifest["source_shape"] != binding["source_shape"]
                or _calibration(manifest.get("calibration")) != _calibration(binding.get("calibration"))
                or not matching_source):
            raise ValueError("来源配置已改变，原除尘数据已保留；请重新进行除尘审核")
        if matching_coordinates:
            return None
        from .repair_geometry import frame_descriptor

        source = frame_descriptor(parent["recipe"], parent, binding["mode"])
        target = frame_descriptor(recipe, manifest, mode)
        if source["output_size"] != [plan["shape"][1], plan["shape"][0]]:
            raise ValueError("原除尘审核画幅与已保存坐标不一致")
        if source == target:
            return None
        return dict(version=1, source=source, target=target)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("当前画面缺少除尘来源证据，请重新审核") from error


def make_plan(receipt_path, manifest_path, mode):
    """Migrate an unchanged historical confirmed receipt without editing it."""
    receipt_path, manifest_path = Path(receipt_path).resolve(), Path(manifest_path).resolve()
    raw, parent_raw = receipt_path.read_bytes(), manifest_path.read_bytes()
    receipt, parent = json.loads(raw), json.loads(parent_raw)
    preview = receipt["preview"]
    return checked_plan(dict(version=1, receipt_path=str(receipt_path),
        receipt_sha256=hashlib.sha256(raw).hexdigest(), mask_path=preview["mask_path"],
        mask_sha256=preview["mask_sha256"], shape=preview["shape"], backend=receipt["backend"],
        binding=dict(mode=mode, inputs=_inputs(parent["inputs"]), calibration=parent.get("calibration"),
                     source_shape=parent["source_shape"], raw_import_identity=_raw_identity(parent),
                     geometry_key=geometry_key(parent["recipe"], parent, mode)),
        parent_manifest=dict(path=str(manifest_path), sha256=hashlib.sha256(parent_raw).hexdigest())))
