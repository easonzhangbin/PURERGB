"""NegPy negative-density film calibration, separate from print paper/rendering.

Profiles are baked into recipes like original NegPy edits. Reading the catalog
never writes the original user's crosstalk directory. This first contract is
Native-only: experimental fits keep their accepted neutral reference.
"""

from dataclasses import replace
import hashlib
from pathlib import Path
import tomllib

import numpy as np

from .engines import ROOT


def generic_film():
    from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX

    return {
        "version": 1,
        "name": "Generic C41",
        "type": "built-in",
        "process": "Color Negative",
        "matrix": list(DEFAULT_CROSSTALK_MATRIX),
        "strength": 0.0,
        "origin": None,
    }


def checked_film(document):
    if (
        not isinstance(document, dict)
        or set(document)
        != {"version", "name", "type", "process", "matrix", "strength", "origin"}
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("不支持的胶片校准配方")
    if (
        not isinstance(document["name"], str)
        or not document["name"].strip()
        or not isinstance(document["type"], str)
    ):
        raise ValueError("胶片校准需名称与来源类型")
    if document["process"] not in ("C41", "Color Negative"):
        raise ValueError("当前三色负片路线只接受 C-41 胶片校准")
    values = document["matrix"]
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 9
        or any(type(v) not in (int, float) for v in values)
    ):
        raise ValueError("胶片矩阵需 9 个有限数值")
    matrix = np.asarray(values, dtype=float).reshape(3, 3)
    # NegPy applies this as a forward transform, without inversion. A large
    # condition number is not a reason to drop an existing measured profile.
    if (
        not np.isfinite(matrix).all()
        or np.max(np.abs(matrix)) > 8
        or np.any(matrix.sum(1) <= 0.05)
    ):
        raise ValueError("胶片矩阵含无效数值或行和过小")
    strength = document["strength"]
    if (
        type(strength) not in (int, float)
        or not np.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("胶片串扰强度需在 0..1")
    origin = document["origin"]
    if origin is not None:
        from .engine_resources import checked_resource

        origin = checked_resource(origin)
    return {
        **document,
        "name": document["name"].strip(),
        "process": "Color Negative",
        "matrix": matrix.reshape(-1).tolist(),
        "strength": float(strength),
        "origin": origin,
    }


def read_film(path):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if len(payload) > 1_000_000:
        raise ValueError("胶片矩阵配置过大")
    value = tomllib.loads(payload.decode("utf-8-sig"))
    rows = value.get("matrix")
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rows)
    ):
        raise ValueError("TOML 需 3×3 matrix")
    return checked_film(
        {
            "version": 1,
            "name": value.get("name", path.stem),
            "type": value.get("type", ""),
            "process": value.get("process", "Color Negative"),
            "matrix": [v for row in rows for v in row],
            "strength": 0.0,
            "origin": {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }
    )


def film_catalog(user_directories=None):
    directories = (
        user_directories
        if user_directories is not None
        else [Path.home() / "Documents/NegPy/crosstalk"]
    )
    result = [generic_film()]
    errors = []
    # As in original NegPy, bundled names take precedence over user duplicates.
    seen = {"Generic C41"}
    for directory in [ROOT / "vendor/negpy/crosstalk", *map(Path, directories)]:
        for path in sorted(directory.glob("*.toml")):
            try:
                item = read_film(path)
                if item["name"] not in seen:
                    result.append(item)
                    seen.add(item["name"])
            except (ValueError, OSError) as error:
                errors.append({"path": str(path), "error": str(error)})
    return result, errors


def film_signature(process):
    return {
        "strength": float(process.crosstalk_strength),
        "matrix": list(process.crosstalk_matrix)
        if process.crosstalk_matrix is not None
        else None,
        "process": str(process.crosstalk_process),
    }


def film_config(config, document):
    film = checked_film(document)
    return replace(
        config,
        process=replace(
            config.process,
            crosstalk_strength=film["strength"],
            crosstalk_matrix=tuple(film["matrix"]),
            crosstalk_profile=film["name"],
            crosstalk_process=film["process"],
        ),
    )


def native_film_context(config, source, roi, density, document):
    if not document or checked_film(document)["strength"] == 0:
        return config, {
            "identity": True,
            "scope": "Native-only negative-density calibration",
        }
    from .density_matrix import negpy_window, percentile_window, fixed_process
    from negpy.features.exposure.normalization import (
        effective_crosstalk_matrix,
        unmix_log_image,
    )
    import negpy_core

    config = film_config(config, document)
    if density == "negpy":
        window = negpy_window(source, config, roi, allow_unmix=True)
    else:
        log = negpy_core.log_density_open(
            np.ascontiguousarray(source, dtype=np.float32)
        )
        unmixed = unmix_log_image(
            log, effective_crosstalk_matrix(config.process, config.process.process_mode)
        )
        transmission = np.power(np.float32(10), unmixed)
        if not np.isfinite(transmission).all():
            raise ValueError("胶片校准产生非有限透射值")
        window = percentile_window(transmission, roi)
        window.metadata["negative_unmix"] = film_signature(config.process)
    config = replace(
        config, process=fixed_process(config.process, window, source.shape)
    )
    from dataclasses import asdict

    return config, {
        "identity": False,
        "scope": "Native-only; recomputed bounds after negative-density unmix, once only",
        "profile": checked_film(document),
        "window": asdict(window),
        "effective_matrix": effective_crosstalk_matrix(
            config.process, config.process.process_mode
        ).tolist(),
        "experimental_reference_changed": False,
    }
