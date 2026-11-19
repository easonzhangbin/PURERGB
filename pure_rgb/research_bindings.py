"""Explicit file-lifetime contracts for new research plans and receipts.

Only code/parameter/receipt roles may explicitly accept a ctime-only observation,
and only after one full SHA comparison. Original frozen records never change.
These checks bind local files, not an authenticated filesystem; the containing
pre-execution plan must independently retain its original bytes/SHA as usual.
"""

import hashlib
import json
import os
from pathlib import Path
import stat


SCHEMA = "pure-rgb-research-file-binding"
AUDIT_SCHEMA = "pure-rgb-research-file-audit"
VERSION = 1
ROLES = ("code", "parameter", "receipt", "raw", "cache", "icc")
CTIME_POLICIES = ("strict", "sha-on-ctime")
_RECOVERABLE_ROLES = ("code", "parameter", "receipt")
_IDENTITY_FIELDS = ("dev", "ino", "size", "mtime_ns", "ctime_ns")
_BINDING_FIELDS = {
    "schema",
    "version",
    "path",
    "role",
    "ctime_policy",
    "sha256",
    "binding_sha256",
    *_IDENTITY_FIELDS,
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    _require(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Research binding requires a lowercase SHA256",
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _policy(role, ctime_policy):
    _require(type(role) is str and role in ROLES, "Unknown explicit research file role")
    _require(
        type(ctime_policy) is str and ctime_policy in CTIME_POLICIES,
        "Research ctime policy must explicitly be strict or sha-on-ctime",
    )
    _require(
        ctime_policy == "strict" or role in _RECOVERABLE_ROLES,
        "RAW/cache/ICC roles require strict identity; ctime recovery is not allowed",
    )


def _resolved(path):
    _require(
        isinstance(path, (str, os.PathLike)) and not isinstance(path, bytes),
        "Research binding path must be a local string/path object",
    )
    try:
        return Path(path).expanduser().resolve(strict=True)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        raise ValueError("Research binding path unavailable or invalid") from error


def _identity(info):
    _require(stat.S_ISREG(info.st_mode), "Research binding requires a regular file")
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _identity_at(path):
    try:
        # The frozen path is already canonical. A later symlink is not accepted
        # as an interchangeable file, even if it resolves to equal bytes.
        return _identity(path.lstat())
    except OSError as error:
        raise ValueError("Bound research file unavailable") from error


def _hash_file(path, expected_identity):
    """Read one complete file, checking its fd and path identities around it."""
    _require(
        _identity_at(path) == expected_identity, "Research file changed before SHA"
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            _require(
                _identity(os.fstat(stream.fileno())) == expected_identity,
                "Opened research file identity differs from frozen candidate",
            )
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            _require(
                _identity(os.fstat(stream.fileno())) == expected_identity,
                "Research file changed during SHA",
            )
    except OSError as error:
        raise ValueError("Bound research file cannot be read") from error
    _require(
        _identity_at(path) == expected_identity, "Research path changed during SHA"
    )
    return digest


def bind_file(path, *, role, ctime_policy="strict"):
    """Freeze one canonical regular file with explicit role/policy and full SHA.

    SHA is read exactly once. All identity fields must remain stable during
    that read. Initial symlink aliases resolve to their canonical target; the
    resulting record binds that target path, not the caller's alias spelling.
    """
    _policy(role, ctime_policy)
    path = _resolved(path)
    identity = _identity_at(path)
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "path": str(path),
        "role": role,
        "ctime_policy": ctime_policy,
        "sha256": _hash_file(path, identity),
        **identity,
    }
    # This seal detects accidental edits to role/path/policy/identity metadata.
    # It is not a signature; the independent plan SHA remains the outer binding.
    result["binding_sha256"] = _digest(result)
    return result


def _checked_binding(binding):
    _require(
        type(binding) is dict and set(binding) == _BINDING_FIELDS,
        "Research binding has missing/unknown fields",
    )
    _require(
        binding["schema"] == SCHEMA
        and type(binding["version"]) is int
        and binding["version"] == VERSION,
        "Unsupported research binding schema/version",
    )
    _policy(binding["role"], binding["ctime_policy"])
    for key in ("sha256", "binding_sha256"):
        _sha(binding[key])
    _require(
        all(type(binding[key]) is int for key in _IDENTITY_FIELDS)
        and all(binding[key] >= 0 for key in ("dev", "ino", "size")),
        "Research binding identity fields must be explicit integers",
    )
    _require(type(binding["path"]) is str, "Frozen research path must be a string")
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    _require(
        _digest(payload) == binding["binding_sha256"],
        "Frozen research metadata changed",
    )
    path = _resolved(binding["path"])
    _require(str(path) == binding["path"], "Frozen research canonical path changed")
    return path, {key: binding[key] for key in _IDENTITY_FIELDS}


def verify_file(binding, *, full_content=False):
    """Audit one frozen file without modifying its original binding.

    Matching identity permits a fast audit with no byte read. Stage-final
    audits must request full_content=True, which hashes the complete eligible
    file exactly once. Invalid schema/path/non-ctime identity fails before a
    potentially unrelated file is read. An allowed ctime-only change always
    requires that one full SHA, even when full_content=False.
    """
    _require(type(full_content) is bool, "full_content must explicitly be boolean")
    path, frozen = _checked_binding(binding)
    observed = _identity_at(path)
    changed = [key for key in _IDENTITY_FIELDS if observed[key] != frozen[key]]
    _require(
        not changed or changed == ["ctime_ns"],
        "Bound research file path/size/mtime/device/inode changed",
    )
    ctime_only = changed == ["ctime_ns"]
    _require(
        not ctime_only or binding["ctime_policy"] == "sha-on-ctime",
        "Bound research file ctime changed under strict policy",
    )
    need_hash = full_content or ctime_only
    digest = _hash_file(path, observed) if need_hash else None
    _require(
        not need_hash or digest == binding["sha256"],
        "Bound research file content SHA changed",
    )
    observation = (
        {
            "kind": "ctime-only-change",
            "frozen_ctime_ns": frozen["ctime_ns"],
            "observed_ctime_ns": observed["ctime_ns"],
            "accepted_by": "sha-on-ctime",
            "full_sha256": digest,
            "frozen_binding_updated": False,
        }
        if ctime_only
        else None
    )
    return {
        "schema": AUDIT_SCHEMA,
        "version": VERSION,
        "status": "verified",
        "path": binding["path"],
        "role": binding["role"],
        "ctime_policy": binding["ctime_policy"],
        "binding_sha256": binding["binding_sha256"],
        "observed_identity": observed,
        "full_content_requested": full_content,
        "content_sha256_verified": need_hash,
        "observed_sha256": digest,
        "hash_reads": int(need_hash),
        "metadata_observation": observation,
        "frozen_binding_updated": False,
    }
