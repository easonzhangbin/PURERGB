"""Inspect Nikon resource capabilities without importing reference/staged code.

Run ``python -m pure_rgb.nikon_resource_probe`` before staging builder research
material. Public CML4 bundle integrity, staged file layout, and acquisition
validation are separate results. File presence never validates a RAW adapter.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct


PUBLIC_COMMIT = "74ffceb79e6e976858d9bffddfeefddba626378f"
# The frozen test app runs its own pure_rgb snapshot while pinned vendor trees
# stay in the original workspace; honor that override. In the repo both roots
# coincide.
DEFAULT_VENDOR = Path(
    os.environ.get("PURE_RGB_TEST_WORKSPACE", Path(__file__).resolve().parents[1])
) / "vendor" / "cool-colors"
SOURCE_CONTRACT = {
    "nikon_cml4.py": (
        20329, "046e6a685ba816cf9b15b772510b4ac415fa856dcabc36869173f4f1454f9f40"
    ),
    "nikon_cml4_oracle.py": (
        15630, "ad73ae2633e9fca2a1f64074a7640d70342ecd2fbc0f2e6c9407cacd0287bd8a"
    ),
    "invert_c41.py": (
        36669, "1884d377ab728b3c204ee2683d03ac8ccc7b3c7bb71a33182008162ec180eea6"
    ),
    "build_builder_luts.py": (
        27619, "8f1eefea7e27e23f16a0ab2ea3d1c65417414da5fccd3bb2d110746a77d0cd6a"
    ),
}
RESOURCE_NAME = "md3-resource-0x100CE578.bin"
RESOURCE_SHA256_PREFIX = "cd934185df496f07"
MAX_STAGED_FILE_BYTES = 1024 * 1024


def _read_record(path, *, size=None, digest=None, maximum=MAX_STAGED_FILE_BYTES):
    """Bound reads, including FIFOs/devices, before hashing or parsing a file."""
    record = {"path": str(path), "status": "missing", "verified": False}
    payload = None
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            record["status"] = "not_regular_file"
            return record, None
        record["bytes"] = info.st_size
        if info.st_size > maximum or (size is not None and info.st_size != size):
            record["status"] = "invalid_size"
            return record, None
        with path.open("rb") as stream:
            payload = stream.read(maximum + 1)
        if len(payload) != info.st_size or len(payload) > maximum:
            record["status"] = "changed_or_oversized"
            return record, None
        actual = hashlib.sha256(payload).hexdigest()
        record.update(sha256=actual, status="present_unverified")
        if digest is not None:
            record.update(verified=actual == digest,
                          status="verified" if actual == digest else "hash_mismatch")
    except FileNotFoundError:
        pass
    except OSError as error:
        record.update(status="unreadable", error=str(error))
    return record, payload


def _literal_contracts(wrapper):
    """Read only literals from the already hash-verified public wrapper."""
    values = {}
    for node in ast.parse(wrapper).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
            value = node.value
        else:
            continue
        if name == "CML_ASSET_SHA256":
            values[name] = ast.literal_eval(value.args[0])
        elif name == "_ASSET_LAYOUT" or name.endswith(("_FILENAME", "_SHA256")):
            values[name] = ast.literal_eval(value)
    return values


def _staged_probe(staged):
    records = {}
    for name in ("nikon_md3_params.py", "nikon_md3_curve.py", RESOURCE_NAME):
        record, payload = _read_record(staged / name, size=1024 if name == RESOURCE_NAME else None)
        records[name] = record
        if payload is None:
            continue
        if name.endswith(".py"):
            try:
                ast.parse(payload)
                record["syntax_valid"] = True
            except (SyntaxError, ValueError, UnicodeError) as error:
                record.update(syntax_valid=False, status="invalid_python_syntax", error=str(error))
        else:
            record.update(
                layout="4 x 32 little-endian float64; master Y then R/G/B X",
                finite=all(math.isfinite(v) for v in struct.unpack("<128d", payload)),
                public_hash_prefix_matches=record["sha256"].startswith(RESOURCE_SHA256_PREFIX),
                full_reference_hash_available=False,
            )
            record["layout_and_prefix_match"] = record["finite"] and record["public_hash_prefix_matches"]
            # A 64-bit hash prefix is the public contract, not a full resource digest.
            if not record["layout_and_prefix_match"]:
                record["status"] = "invalid_resource_contract"

    oracle = staged / "oracle-capture-d"
    descriptor, blob = _read_record(oracle / "analyzer-desc.bin")
    records["oracle-capture-d/analyzer-desc.bin"] = descriptor
    expected_pixels = None
    if blob is not None:
        valid = len(blob) >= 0x58
        if valid:
            width, height, r0, c0, r1, c1 = struct.unpack_from("<6I", blob)
            denominators = struct.unpack_from("<3I", blob, 0x28)
            densities = struct.unpack_from("<3d", blob, 0x40)
            valid = (
                (height, width) == (425, 281)
                and (r0, c0, r1, c1) == (50, 36, 375, 245)
                and all(denominators) and all(map(math.isfinite, densities))
            )
            if valid:
                expected_pixels = width * height * 3 * 2
        descriptor["layout_valid"] = bool(valid)
        if not valid:
            descriptor["status"] = "invalid_descriptor"

    sizes = {"analyzer-pixels.bin": expected_pixels, "builder-args.bin": 3 * 0x44}
    sizes.update({f"builder-preF-{c}.bin": 65536 * 2 for c in "rgb"})
    sizes.update({f"builder-control-{axis}-{c}.bin": None for axis in "xy" for c in "rgb"})
    for name, size in sizes.items():
        record, payload = _read_record(oracle / name, size=size)
        records["oracle-capture-d/" + name] = record
        if payload is not None:
            valid = bool(payload)
            if name.startswith("builder-control-"):
                valid = len(payload) >= 16 and len(payload) % 8 == 0
                if valid:
                    valid = all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", payload))
            elif name == "analyzer-pixels.bin":
                valid = expected_pixels is not None and len(payload) == expected_pixels
            elif name == "builder-args.bin":
                valid = all(
                    math.isfinite(v)
                    for c in range(3)
                    for v in struct.unpack_from("<7d", payload, c * 0x44)
                )
            record["layout_valid"] = bool(valid)
            if not valid:
                record["status"] = "invalid_layout"
    layout_complete = all(
        (r.get("syntax_valid") if name.endswith(".py") else
         r.get("layout_and_prefix_match") if name == RESOURCE_NAME else r.get("layout_valid"))
        for name, r in records.items()
    )
    return {
        "directory": str(staged), "files": records,
        "layout_complete": bool(layout_complete),
        "module_provenance_verified": False,
        "oracle_provenance_verified": False,
        "oracle_replay_performed": False,
        "note": "AST and binary-layout inspection only; staged Python is never imported or executed",
    }


def probe_resources(vendor=DEFAULT_VENDOR, *, staged=None):
    vendor = Path(vendor).resolve()
    staged = Path(staged).resolve() if staged is not None else vendor / ".refstage"
    sources, wrapper = {}, None
    for name, (size, digest) in SOURCE_CONTRACT.items():
        record, payload = _read_record(vendor / name, size=size, digest=digest)
        sources[name] = record
        if name == "nikon_cml4.py" and record["verified"]:
            wrapper = payload
    assets = {}
    if wrapper is not None:
        contract = _literal_contracts(wrapper)
        base = vendor / "assets" / "nikon_cml4"
        for name, digest in contract["CML_ASSET_SHA256"].items():
            size = math.prod(contract["_ASSET_LAYOUT"][name][1]) * 2
            assets[name] = _read_record(base / name, size=size, digest=digest)[0]
        for key, size in (("VALIDATION_RECEIPT", 4014), ("FIXED_LUT", 131072), ("NIKON_PROFILE", 492)):
            name = contract[key + "_FILENAME"]
            assets[name] = _read_record(base / name, size=size, digest=contract[key + "_SHA256"])[0]
    public_verified = bool(assets) and all(r["verified"] for r in sources.values()) and all(r["verified"] for r in assets.values())
    return {
        "schema": "pure-rgb-nikon-resource-capabilities", "version": 1,
        "public_reference_commit": PUBLIC_COMMIT, "vendor_directory": str(vendor),
        "public_source_files": sources, "public_cml4_assets": assets,
        "staged_builder_material": _staged_probe(staged),
        "capabilities": {
            "public_downstream_resource_integrity_verified": public_verified,
            "downstream_numerical_test_performed": False,
            "fresh_acquisition_builder_generation_verified": False,
            "acquisition_binding_verified": False,
            "camera_to_ls5000_adapter_verified": False,
            "factory_equivalent_rw2_route_available": False,
        },
        "scope": "Pinned downstream resources are independently available; staged builder layout does not prove acquisition binding or camera/scanner domain equivalence",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--staged", type=Path, help="Optional staged builder research directory")
    parser.add_argument("--output", type=Path, help="Write a new JSON receipt; refuse overwrite")
    args = parser.parse_args(argv)
    report = probe_resources(args.vendor, staged=args.staged)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0 if report["capabilities"]["public_downstream_resource_integrity_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
