"""Metadata/file intake for future C06/C07/C08 physical reference bundles.

No image decoding, kernels, CMS, fitting or physical provenance authentication.
Passing checks means internally consistent, file-bound *claims*, never accuracy.
Unknown measurements must be explicit None and keep input_complete false.
"""

from copy import deepcopy
import math

from pure_rgb import research_bindings


SCHEMA = "pure-rgb-paired-reference-bundle"
VERSION = 1
CHANNELS = ("R", "G", "B")
_CAMERA = {
    "camera_raw",
    "dark",
    "flat",
    "exposure_step",
    "stable_reference",
    "film_base",
}
_DOMAINS = {
    **{role: "camera-sensor" for role in _CAMERA},
    "camera_preinversion": "camera-linear-negative",
    "nikon_prescan": "nikon-prescan-device",
    "nikon_final_codes": "nikon-final-device",
    "flex_3f": "flex-native-3f",
    "flex_device_codes": "flex-device",
    "original_tiff": "icc-positive",
}
_META = {"settings", "icc", "transmission_reference", "led_spectrum", "sensor_response"}
_DERIVED = {"camera_preinversion", "flex_device_codes", "original_tiff"}
_IDENTITY = ("stock", "roll_id", "process_id", "frame_mark")
_TOP = {
    "schema",
    "version",
    "bundle_id",
    "frames",
    "artifacts",
    "camera_setup",
    "nikon_pairs",
    "flex_pairs",
    "acquisition",
    "transmission_reference",
    "model",
}


def _require(ok, code, message):
    if not ok:
        raise ValueError(f"{code}: {message}")


def _object(value, keys, code):
    _require(
        type(value) is dict and set(value) == set(keys), code, "missing/unknown fields"
    )


def _text(value, code):
    _require(
        type(value) is str and value.strip() == value and bool(value),
        code,
        "expected explicit nonempty string",
    )


def _number(value, code, *, positive=False):
    _require(
        type(value) in (int, float) and math.isfinite(value),
        code,
        "expected finite number",
    )
    _require(
        value > 0 if positive else value >= 0, code, "number outside allowed range"
    )


def _known(value, label, missing, validator=None):
    if value is None:
        missing.append(label)
    elif validator is not None:
        validator(value)


def _strings(value, code, *, nonempty=True):
    _require(
        type(value) is list and (bool(value) or not nonempty), code, "expected list"
    )
    for item in value:
        _text(item, code)
    _require(len(set(value)) == len(value), code, "duplicate identifiers")


def _binding_role(role):
    if role in _CAMERA or role == "flex_3f":
        return "raw"
    if role == "icc":
        return "icc"
    if role in _META:
        return "receipt"
    return "cache"


def _check_setup(setup, missing):
    def checked_temperature(value):
        _require(
            type(value) in (int, float) and math.isfinite(value),
            "capture",
            "invalid temperature",
        )

    def checked_text(value):
        _text(value, "capture")

    fields = {
        "id",
        "camera_model",
        "camera_serial",
        "lens",
        "aperture",
        "focus",
        "magnification",
        "iso",
        "sensor_mode",
        "temperature_c",
        "leds",
    }
    _object(setup, fields, "capture")
    _text(setup["id"], "capture")
    for field in fields - {"id", "leds"}:
        validator = (
            (lambda v: _number(v, "capture", positive=True))
            if field in {"aperture", "magnification", "iso"}
            else None
        )
        if field == "temperature_c":
            validator = checked_temperature
        elif validator is None:
            validator = checked_text
        _known(setup[field], "camera_setup." + field, missing, validator)
    _object(setup["leds"], CHANNELS, "capture")
    for channel, led in setup["leds"].items():
        _object(led, {"id", "current_ma", "warmup_s"}, "capture")
        for field, value in led.items():
            validator = (
                (lambda v: _text(v, "capture"))
                if field == "id"
                else (lambda v: _number(v, "capture", positive=True))
            )
            _known(value, f"camera_setup.leds.{channel}.{field}", missing, validator)


def _check_encoding(artifact, missing):
    role, coordinate, encoding = (
        artifact[k] for k in ("role", "coordinate", "encoding")
    )
    if role in _META:
        _require(
            coordinate is None and encoding is None and artifact["capture"] is None,
            "domain",
            "metadata/resources cannot be pixel observations",
        )
        return
    _object(
        coordinate, {"domain", "width", "height", "orientation", "mapping_id"}, "domain"
    )
    _require(
        coordinate["domain"] == _DOMAINS[role],
        "domain",
        "role has wrong coordinate/signal domain",
    )
    for field in ("width", "height"):
        _known(
            coordinate[field],
            artifact["id"] + "." + field,
            missing,
            lambda v: _require(
                type(v) is int and v > 0, "domain", "invalid dimensions"
            ),
        )
    for field in ("orientation", "mapping_id"):
        _known(
            coordinate[field],
            artifact["id"] + "." + field,
            missing,
            lambda v: _text(v, "domain"),
        )
    _object(
        encoding,
        {
            "kind",
            "container_bits",
            "effective_bits",
            "black",
            "white",
            "channels",
            "transfer",
        },
        "encoding",
    )
    expected = (
        "linear-integer"
        if role in _CAMERA
        else {
            "camera_preinversion": "linear-float",
            "flex_3f": "native-container",
            "original_tiff": "icc-integer",
        }.get(role, "device-integer")
    )
    _require(encoding["kind"] == expected, "encoding", "role has wrong code encoding")
    if role == "flex_3f":
        _require(
            all(
                encoding[k] is None
                for k in (
                    "container_bits",
                    "effective_bits",
                    "black",
                    "white",
                    "channels",
                    "transfer",
                )
            ),
            "encoding",
            "3F container cannot claim decoded code interpretation",
        )
        return
    for field in ("container_bits", "effective_bits"):
        _known(
            encoding[field],
            artifact["id"] + "." + field,
            missing,
            lambda v: _require(
                type(v) is int and 1 <= v <= 64, "encoding", "invalid code depth"
            ),
        )
    bits, effective = encoding["container_bits"], encoding["effective_bits"]
    if bits is not None and effective is not None:
        _require(effective <= bits, "encoding", "effective bits exceed container")
    for field in ("black", "white"):
        _known(
            encoding[field],
            artifact["id"] + "." + field,
            missing,
            lambda v: _number(v, "encoding"),
        )
    black, white = encoding["black"], encoding["white"]
    if black is not None and white is not None:
        _require(white > black, "encoding", "white must exceed black")
        if bits is not None and expected != "linear-float":
            _require(white <= 2**bits - 1, "encoding", "code exceeds container")
    _require(
        encoding["channels"] == list(CHANNELS),
        "encoding",
        "explicit RGB order required",
    )
    transfer = (
        "linear"
        if expected.startswith("linear")
        else "icc"
        if role == "original_tiff"
        else "device"
    )
    _require(
        encoding["transfer"] == transfer, "encoding", "wrong transfer interpretation"
    )


def _check_capture(artifact, setup, missing):
    if artifact["role"] in _META:
        return
    capture = artifact["capture"]
    _object(capture, {"setup_id", "exposure_s", "channel"}, "capture")
    _text(capture["setup_id"], "capture")
    _require(
        capture["channel"] in (*CHANNELS, "RGB"),
        "capture",
        "unknown illumination channel",
    )
    if artifact["role"] in _CAMERA or artifact["role"] == "camera_preinversion":
        _require(capture["setup_id"] == setup["id"], "capture", "camera setup differs")
    if artifact["role"] in _CAMERA:
        _require(
            capture["channel"] in CHANNELS, "capture", "separate LED capture required"
        )
    else:
        _require(capture["channel"] == "RGB", "capture", "RGB observation required")
    value = capture["exposure_s"]
    if artifact["role"] in _CAMERA:
        _known(
            value,
            artifact["id"] + ".exposure_s",
            missing,
            lambda v: _number(v, "capture", positive=True),
        )
    else:
        _require(
            type(value) is list and len(value) == 3,
            "capture",
            "record RGB exposure linkage",
        )
        for channel, exposure in zip(CHANNELS, value):
            _known(
                exposure,
                artifact["id"] + ".exposure_s." + channel,
                missing,
                lambda v: _number(v, "capture", positive=True),
            )


def _check_lineage(artifacts):
    visiting, visited = set(), set()

    def visit(key):
        _require(key not in visiting, "lineage", "cycle")
        if key in visited:
            return
        visiting.add(key)
        item = artifacts[key]
        for parent_id in item["parents"]:
            _require(parent_id in artifacts, "lineage", "missing parent")
            parent = artifacts[parent_id]
            _require(
                parent["frame_id"] in (None, item["frame_id"]),
                "lineage",
                "parent belongs to another physical frame/split",
            )
            visit(parent_id)
        visiting.remove(key)
        visited.add(key)

    for key in artifacts:
        visit(key)


def _check_pairs(pairs, route, frames, artifacts, missing):
    _require(type(pairs) is list, "pair", "expected pair list")
    covered = set()
    scanner_roles = (
        {"prescan_id": "nikon_prescan", "final_codes_id": "nikon_final_codes"}
        if route == "nikon"
        else {"three_f_id": "flex_3f", "device_codes_id": "flex_device_codes"}
    )
    fields = {
        "frame_id",
        "camera_raw_ids",
        "camera_preinversion_id",
        "settings_id",
        "original_tiff_id",
        "icc_id",
        *scanner_roles,
    }
    for pair in pairs:
        _object(pair, fields, "pair")
        frame_id = pair["frame_id"]
        _text(frame_id, "pair")
        _require(
            frame_id in frames and frame_id not in covered,
            "pair",
            "unknown/repeated paired physical frame",
        )
        covered.add(frame_id)
        raw_ids = pair["camera_raw_ids"]
        _strings(raw_ids, "pair")
        _require(len(raw_ids) == 3, "pair", "three separate LED RAWs required")

        def require_role(key, role, *, shared=False):
            _text(key, "pair")
            _require(
                key in artifacts and artifacts[key]["role"] == role,
                "pair",
                "missing/wrong artifact role",
            )
            item = artifacts[key]
            _require(
                item["frame_id"] == (None if shared else frame_id),
                "pair",
                "artifact not paired to same physical frame",
            )
            return item

        raw = [require_role(key, "camera_raw") for key in raw_ids]
        _require(
            [r["capture"]["channel"] for r in raw] == list(CHANNELS),
            "pair",
            "camera IDs must be R/G/B captures in order",
        )
        pre = require_role(pair["camera_preinversion_id"], "camera_preinversion")
        _require(
            set(pre["parents"]) == set(raw_ids),
            "lineage",
            "pre-inversion must link the three original RAWs",
        )
        _require(
            pre["capture"]["exposure_s"] == [r["capture"]["exposure_s"] for r in raw],
            "pair",
            "camera exposure linkage differs",
        )
        scanner = {
            key: require_role(pair[key], role) for key, role in scanner_roles.items()
        }
        settings = require_role(pair["settings_id"], "settings")["details"]
        settings_fields = {
            "route",
            "device_model",
            "device_serial",
            "app_version",
            "cmm",
            "session_id",
            "exposure_owner",
            "exposure_rgb_s",
            "film_mode",
            "film_profile",
            "automatic_settings",
            "preferences",
            "descriptor",
            "output_icc_id",
        }
        _object(settings, settings_fields, "pair")
        _require(
            settings["route"] == route and settings["output_icc_id"] == pair["icc_id"],
            "pair",
            "settings route/output profile differs",
        )
        for field in settings_fields - {"route", "output_icc_id", "exposure_rgb_s"}:
            _known(
                settings[field],
                pair["settings_id"] + "." + field,
                missing,
                lambda v: _text(v, "pair"),
            )
        stages = {"prescan", "final"} if route == "nikon" else {"final"}
        _object(settings["exposure_rgb_s"], stages, "pair")
        for stage, exposures in settings["exposure_rgb_s"].items():
            _require(
                type(exposures) is list and len(exposures) == 3,
                "pair",
                "settings need three exposure values",
            )
            for exposure in exposures:
                _known(
                    exposure,
                    pair["settings_id"] + ".exposure." + stage,
                    missing,
                    lambda v: _number(v, "pair", positive=True),
                )
        for key, item in scanner.items():
            stage = "prescan" if key == "prescan_id" else "final"
            _require(
                item["capture"]["setup_id"] == settings["session_id"],
                "pair",
                "scanner session differs",
            )
            _require(
                item["capture"]["exposure_s"] == settings["exposure_rgb_s"][stage],
                "pair",
                "scanner exposure/settings linkage differs",
            )
        require_role(pair["icc_id"], "icc", shared=True)
        original = require_role(pair["original_tiff_id"], "original_tiff")
        _require(
            original["capture"]["setup_id"] == settings["session_id"]
            and original["capture"]["exposure_s"]
            == settings["exposure_rgb_s"]["final"],
            "pair",
            "original output lacks final acquisition/session linkage",
        )
        input_id = (
            pair["final_codes_id"] if route == "nikon" else pair["device_codes_id"]
        )
        _require(
            set(original["parents"]) == {input_id, pair["settings_id"], pair["icc_id"]},
            "lineage",
            "original TIFF lacks code/settings/profile lineage",
        )
        _object(
            original["details"],
            {"original_application", "version", "cmm", "output_icc_id"},
            "pair",
        )
        details = original["details"]
        for field in details:
            _known(
                details[field],
                original["id"] + "." + field,
                missing,
                lambda v: _text(v, "pair"),
            )
        _require(
            details["output_icc_id"] == pair["icc_id"]
            and details["version"] == settings["app_version"]
            and details["cmm"] == settings["cmm"],
            "pair",
            "original output application/profile metadata differs",
        )
        if route == "flex":
            _require(
                scanner["device_codes_id"]["parents"] == [pair["three_f_id"]],
                "lineage",
                "decoded Flex codes must identify original 3F parent",
            )
    _require(
        covered == set(frames),
        "pair",
        "each calibration and heldout frame needs both routes",
    )


def _check_acquisition(document, frames, artifacts, missing):
    _object(document, {"per_led", "film_base_ids"}, "acquisition")
    _object(document["per_led"], CHANNELS, "acquisition")
    for channel, row in document["per_led"].items():
        _object(
            row,
            {"dark_id", "flat_id", "exposure_step_ids", "stable_reference_id"},
            "acquisition",
        )

        def observation(key, role):
            _require(
                key in artifacts and artifacts[key]["role"] == role,
                "acquisition",
                "missing/wrong calibration observation",
            )
            item = artifacts[key]
            _require(
                item["frame_id"] is None and item["capture"]["channel"] == channel,
                "acquisition",
                "unmatched LED calibration",
            )
            return item

        dark = observation(row["dark_id"], "dark")
        flat = observation(row["flat_id"], "flat")
        stable = observation(row["stable_reference_id"], "stable_reference")
        _require(
            dark["capture"]["exposure_s"] == flat["capture"]["exposure_s"],
            "acquisition",
            "dark and flat need matched exposure",
        )
        _strings(row["exposure_step_ids"], "acquisition")
        _require(
            len(row["exposure_step_ids"]) >= 3,
            "acquisition",
            "at least three measured exposure steps required",
        )
        ladder = [observation(key, "exposure_step") for key in row["exposure_step_ids"]]
        times = [item["capture"]["exposure_s"] for item in ladder]
        if all(time is not None for time in times):
            _require(
                all(a < b for a, b in zip(times, times[1:])),
                "acquisition",
                "ladder exposure must strictly increase",
            )
        for item in [dark, flat, stable, *ladder]:
            _object(
                item["details"],
                {"observation_id", "stable_reference_id", "measured_response_id"},
                "acquisition",
            )
            for field, value in item["details"].items():
                _known(
                    value,
                    item["id"] + "." + field,
                    missing,
                    lambda v: _text(v, "acquisition"),
                )
            _require(
                item["details"]["stable_reference_id"] == stable["id"],
                "acquisition",
                "stable reference linkage differs",
            )
            _require(
                item["details"]["measured_response_id"] in (None, item["id"]),
                "acquisition",
                "response evidence must identify this bound RAW observation; no response model is evaluated",
            )
    _strings(document["film_base_ids"], "acquisition")
    covered = set()
    for key in document["film_base_ids"]:
        _require(
            key in artifacts and artifacts[key]["role"] == "film_base",
            "acquisition",
            "missing processed film base",
        )
        item = artifacts[key]
        _require(
            item["frame_id"] in frames,
            "acquisition",
            "film base belongs to unknown physical film frame",
        )
        _require(
            item["details"] == frames[item["frame_id"]]["film"],
            "acquisition",
            "film base stock/process/frame differs",
        )
        identity = (item["frame_id"], item["capture"]["channel"])
        _require(identity not in covered, "acquisition", "repeated frame/LED film base")
        covered.add(identity)
    _require(
        covered == {(key, channel) for key in frames for channel in CHANNELS},
        "acquisition",
        "per-LED processed base required for both frame splits",
    )


def _check_reference(reference, frames, artifacts, pairs, missing):
    _object(
        reference,
        {"kind", "target_id", "reference_id", "conditions", "patch_sets"},
        "reference",
    )
    _require(
        type(reference["kind"]) is str
        and reference["kind"] in {"certified-transmission", "measured-transmission"},
        "reference",
        "reflection chart or film Passport cannot substitute transmission measurements",
    )
    _known(
        reference["target_id"],
        "transmission_reference.target_id",
        missing,
        lambda v: _text(v, "reference"),
    )
    key = reference["reference_id"]
    _text(key, "reference")
    _require(
        key in artifacts and artifacts[key]["role"] == "transmission_reference",
        "reference",
        "missing bound transmission reference",
    )
    _object(
        reference["conditions"],
        {"instrument", "measurement_mode", "illumination", "observer", "version"},
        "reference",
    )
    for field, value in reference["conditions"].items():
        _known(
            value,
            "reference.conditions." + field,
            missing,
            lambda v: _text(v, "reference"),
        )
    _object(reference["patch_sets"], {"calibration", "heldout"}, "reference")
    covered, patch_ids = set(), set()
    for split, rows in reference["patch_sets"].items():
        _require(
            type(rows) is list and bool(rows),
            "reference",
            "missing transmission patch sets",
        )
        for row in rows:
            _object(
                row,
                {"frame_id", "gray_steps", "color_patches", "measurement_evidence_id"},
                "reference",
            )
            frame_id = row["frame_id"]
            _text(frame_id, "reference")
            _require(
                frame_id in frames
                and frames[frame_id]["split"] == split
                and frame_id not in covered,
                "split",
                "reference patches reuse calibration physical frame as heldout",
            )
            covered.add(frame_id)
            _require(
                row["measurement_evidence_id"] == key,
                "reference",
                "patch measurements lack bound reference evidence",
            )
            for kind in ("gray_steps", "color_patches"):
                patches = row[kind]
                _require(
                    type(patches) is list and len(patches) >= 3,
                    "reference",
                    "three gray steps and three colors required per frame",
                )
                for patch in patches:
                    _object(
                        patch,
                        {"id", "quantity", "values", "reference_record_key", "mapping"},
                        "reference",
                    )
                    _text(patch["id"], "reference")
                    _require(
                        patch["id"] not in patch_ids,
                        "reference",
                        "repeated physical patch identity",
                    )
                    patch_ids.add(patch["id"])
                    _require(
                        patch["quantity"] == "transmittance-rgb",
                        "reference",
                        "unknown/unmeasured reference is not a measurement",
                    )
                    _known(
                        patch["reference_record_key"],
                        "reference.patch." + patch["id"] + ".reference_record_key",
                        missing,
                        lambda v: _text(v, "reference"),
                    )
                    mapping_roles = {
                        "camera_preinversion": "camera_preinversion_id",
                        "nikon_prescan": "prescan_id",
                        "nikon_final_codes": "final_codes_id",
                        "flex_device_codes": "device_codes_id",
                    }
                    _object(patch["mapping"], mapping_roles, "reference")
                    for role, pair_key in mapping_roles.items():
                        roi = patch["mapping"][role]
                        if roi is None:
                            missing.append(
                                "reference.patch." + patch["id"] + ".mapping." + role
                            )
                            continue
                        _require(
                            type(roi) is list
                            and len(roi) == 4
                            and all(type(v) is int for v in roi),
                            "reference",
                            "patch ROI must contain four integer coordinates",
                        )
                        pair = pairs[
                            "flex" if role == "flex_device_codes" else "nikon"
                        ][frame_id]
                        coordinate = artifacts[pair[pair_key]]["coordinate"]
                        x0, y0, x1, y1 = roi
                        _require(
                            0 <= x0 < x1 and 0 <= y0 < y1,
                            "reference",
                            "patch ROI has invalid ordering",
                        )
                        if (
                            coordinate["width"] is not None
                            and coordinate["height"] is not None
                        ):
                            _require(
                                x1 <= coordinate["width"]
                                and y1 <= coordinate["height"],
                                "reference",
                                "patch ROI outside claimed coordinate domain",
                            )
                    values = patch["values"]
                    if values is None:
                        missing.append("reference.patch." + patch["id"])
                    else:
                        _require(
                            type(values) is list and len(values) == 3,
                            "reference",
                            "transmission RGB values required",
                        )
                        for value in values:
                            _number(value, "reference")
                            _require(
                                value <= 1, "reference", "transmittance outside 0..1"
                            )
    _require(
        covered == set(frames),
        "reference",
        "independent reference measurements needed for both frame splits",
    )


def check_bundle(document, *, verify_files=True):
    """Check a strict intake manifest; full SHA/identity verification by default.

    File checks do not parse payloads or establish that frame, exposure, measured
    values, application, ICC use, coordinates, or original-output claims are true.
    Structural-only checks are explicit and cannot report input_complete=True.
    Caller retains/freeze-binds the manifest itself before later evaluation.
    """
    _require(type(verify_files) is bool, "schema", "verify_files must be boolean")
    _object(document, _TOP, "schema")
    _require(
        document["schema"] == SCHEMA
        and type(document["version"]) is int
        and document["version"] == VERSION,
        "schema",
        "unsupported schema/version",
    )
    _text(document["bundle_id"], "schema")
    missing = []
    _check_setup(document["camera_setup"], missing)
    _require(
        type(document["frames"]) is list and len(document["frames"]) >= 2,
        "split",
        "independent calibration and heldout physical frames required",
    )
    frames, physical = {}, set()
    for frame in document["frames"]:
        _object(frame, {"id", "film", "split"}, "frame")
        _text(frame["id"], "frame")
        _require(frame["id"] not in frames, "frame", "duplicate physical frame ID")
        _object(frame["film"], _IDENTITY, "frame")
        for value in frame["film"].values():
            _text(value, "frame")
        identity = tuple(frame["film"][k] for k in _IDENTITY)
        _require(
            identity not in physical,
            "frame",
            "same physical film identity has multiple IDs",
        )
        _require(
            type(frame["split"]) is str
            and frame["split"] in {"calibration", "heldout"},
            "split",
            "unknown split",
        )
        physical.add(identity)
        frames[frame["id"]] = frame
    _require(
        {f["split"] for f in frames.values()} == {"calibration", "heldout"},
        "split",
        "calibration and heldout must both exist",
    )
    _require(
        type(document["artifacts"]) is list and bool(document["artifacts"]),
        "artifact",
        "missing artifacts",
    )
    artifacts, paths, identities, frame_hashes = {}, set(), set(), {}
    for item in document["artifacts"]:
        _object(
            item,
            {
                "id",
                "role",
                "binding",
                "frame_id",
                "parents",
                "provenance",
                "coordinate",
                "encoding",
                "capture",
                "details",
            },
            "artifact",
        )
        _text(item["id"], "artifact")
        _require(item["id"] not in artifacts, "artifact", "duplicate artifact ID")
        role = item["role"]
        _text(role, "role")
        _require(role in {*_DOMAINS, *_META}, "role", "unknown artifact role")
        _require(
            item["frame_id"] is None
            or (type(item["frame_id"]) is str and item["frame_id"] in frames),
            "frame",
            "unknown artifact physical frame",
        )
        _strings(item["parents"], "lineage", nonempty=False)
        expected_provenance = (
            "derived"
            if role in _DERIVED
            else "resource"
            if role == "icc"
            else "measurement"
            if role in {"transmission_reference", "led_spectrum", "sensor_response"}
            else "capture"
        )
        _require(
            item["provenance"] == expected_provenance,
            "lineage",
            "role has wrong provenance",
        )
        _require(
            bool(item["parents"]) if role in _DERIVED else not item["parents"],
            "lineage",
            "capture/resource cannot be relabeled derivative, derivatives require parents",
        )
        _require(
            type(item["details"]) is dict,
            "artifact",
            "explicit metadata details object required",
        )
        binding = item["binding"]
        _require(
            type(binding) is dict
            and binding.get("role") == _binding_role(role)
            and binding.get("ctime_policy") == "strict",
            "binding",
            "artifact binding requires correct role and strict identity",
        )
        # Even structural-only mode checks the complete sealed binding schema.
        try:
            research_bindings._checked_binding(binding)
        except (ValueError, OSError) as error:
            raise ValueError(f"binding: {error}") from error
        identity = tuple(binding[k] for k in ("dev", "ino"))
        _require(
            binding["path"] not in paths and identity not in identities,
            "binding",
            "one file cannot masquerade as multiple artifacts",
        )
        paths.add(binding["path"])
        identities.add(identity)
        if item["frame_id"] is not None and role not in _META:
            digest = binding["sha256"]
            prior_frame = frame_hashes.setdefault(digest, item["frame_id"])
            _require(
                prior_frame == item["frame_id"],
                "split",
                "equal bytes relabeled as independent physical frames",
            )
        _check_encoding(item, missing)
        _check_capture(item, document["camera_setup"], missing)
        artifacts[item["id"]] = item
    camera_observations = [a for a in artifacts.values() if a["role"] in _CAMERA]
    if camera_observations:
        reference = camera_observations[0]
        for item in camera_observations[1:]:
            for field in ("width", "height", "orientation", "mapping_id"):
                a, b = reference["coordinate"][field], item["coordinate"][field]
                if a is not None and b is not None:
                    _require(
                        a == b,
                        "domain",
                        "camera/raw/calibration observations use different sensor grids",
                    )
            for field, a in reference["encoding"].items():
                b = item["encoding"][field]
                if a is not None and b is not None:
                    _require(
                        a == b,
                        "encoding",
                        "camera/raw/calibration code interpretations differ",
                    )
    _check_lineage(artifacts)
    for route in ("nikon", "flex"):
        _check_pairs(document[route + "_pairs"], route, frames, artifacts, missing)
    _check_acquisition(document["acquisition"], frames, artifacts, missing)
    pairs = {
        route: {p["frame_id"]: p for p in document[route + "_pairs"]}
        for route in ("nikon", "flex")
    }
    _check_reference(
        document["transmission_reference"], frames, artifacts, pairs, missing
    )
    model = document["model"]
    _object(model, {"family", "independent_constraint_ids"}, "schema")
    _text(model["family"], "schema")
    _require(
        model["family"] in {"empirical-paired-rgb", "spectral"},
        "schema",
        "unsupported model family",
    )
    _strings(model["independent_constraint_ids"], "reference", nonempty=False)
    constraints = []
    for key in model["independent_constraint_ids"]:
        _require(
            key in artifacts
            and artifacts[key]["role"] in {"led_spectrum", "sensor_response"},
            "reference",
            "spectral constraint requires independent measured resource",
        )
        item = artifacts[key]
        _require(
            item["frame_id"] is None,
            "reference",
            "spectral constraint cannot be a transformed calibration frame",
        )
        constraints.append(item)
    spectral_complete = {a["role"] for a in constraints} == {
        "led_spectrum",
        "sensor_response",
    } and len([a for a in constraints if a["role"] == "led_spectrum"]) == 3
    if model["family"] == "spectral":
        # RGB transmittance values alone do not identify spectra, even with LED
        # and sensor constraints; measured film spectral transmission is needed.
        if not spectral_complete:
            missing.append("model.independent_LED_spectra_and_camera_response")
        missing.append(
            "model.independent_film_spectral_transmission_not_supported_by_RGB_schema"
        )
    audits = []
    if verify_files:
        for item in artifacts.values():
            try:
                audits.append(
                    research_bindings.verify_file(item["binding"], full_content=True)
                )
            except (ValueError, OSError) as error:
                raise ValueError(f"binding: {error}") from error
    else:
        missing.append("current_file_identity_and_full_SHA_not_verified")
    return {
        "schema": "pure-rgb-paired-reference-intake-report",
        "version": VERSION,
        "bundle_id": document["bundle_id"],
        "input_complete": not missing,
        "missing_measurements": sorted(set(missing)),
        "files_verified_full_sha": len(audits),
        "frame_splits": {
            split: sorted(k for k, f in frames.items() if f["split"] == split)
            for split in ("calibration", "heldout")
        },
        "physical_provenance_verified": False,
        "physical_accuracy_evaluated": False,
        "factory_equivalence_proven": False,
        "spectral_response_identified": False,
        "interpretation": "Complete means internally consistent supplied claims and current file identity/full SHA. Coordinates, measurement values, physical frame identity and original-application origin are unauthenticated claims. No image or color kernel was evaluated.",
    }


def capture_requirements():
    """Smallest required capture manifests; None means not yet measured/supplied.

    These are intake requirements, deliberately not valid/completed bundles.
    Both scanner routes need calibration and independent heldout physical frames.
    """
    shared = {
        "physical_frames": {"calibration": None, "heldout": None},
        "physical_frame_identity": list(_IDENTITY),
        "file_binding": "canonical regular file, strict identity, complete SHA256; manually supplied provenance remains a claim",
        "camera": {
            "same_frame_R_G_B_RAW": None,
            "linear_preinversion_and_exact_parent_ids": None,
            "exposure_and_setup": None,
            "coordinate_mapping_and_code_encoding": None,
        },
    }
    return {
        "schema": "pure-rgb-minimal-capture-requirements",
        "version": VERSION,
        "input_complete": False,
        "NIKON_PAIRED_FRAME_V1": {
            **deepcopy(shared),
            "scanner": {
                "same_frame_prescan_codes": None,
                "same_frame_final_preinversion_codes": None,
                "exposure_owner_descriptor_prescan_final_link": None,
                "device_serial_and_original_settings_preferences": None,
                "original_application_TIFF_and_actual_output_ICC": None,
                "application_version_CMM_and_session": None,
            },
        },
        "FLEX_PAIRED_3F_V1": {
            **deepcopy(shared),
            "scanner": {
                "same_frame_native_3F": None,
                "device_codes_and_extraction_lineage": None,
                "device_serial_exposure_film_profile_and_auto_settings": None,
                "original_preferences_and_gray_CMS_state": None,
                "original_application_TIFF_and_actual_output_ICC": None,
                "application_version_CMM_and_session": None,
            },
        },
        "SHARED_CAPTURE_CALIBRATION_V1": {
            "matched_setup": [
                "aperture",
                "focus",
                "magnification",
                "ISO",
                "sensor_mode",
                "temperature",
                "per_LED_id_current_warmup",
            ],
            "per_LED": {
                channel: {
                    "dark_bias": None,
                    "empty_carrier_flat": None,
                    "exposure_ladder_at_least_three_steps": None,
                    "stable_reference_and_observed_response": None,
                }
                for channel in CHANNELS
            },
            "same_processed_film_base_for_each_frame": None,
            "metadata_black_white_are_not_measurements": True,
        },
        "INDEPENDENT_TRANSMISSION_COLOR_REFERENCE_V1": {
            "measured_or_certified_transmission_target_or_same_film_patches": None,
            "bound_reference_version_instrument_mode_illuminant_observer": None,
            "per_patch_reference_record_key_and_camera_scanner_ROIs": None,
            "gray_steps_at_least_three_per_frame": None,
            "color_patches_at_least_three_per_frame": None,
            "independent_calibration_and_heldout_frames": None,
            "reflection_Passport_or_its_film_record_is_sufficient": False,
            "spectral_model_additionally_requires": [
                "independent per_LED spectra",
                "camera spectral response",
                "film patch spectral transmission",
            ],
        },
    }
