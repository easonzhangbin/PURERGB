"""Explicit SDK C41 log-density domain for masked Native processing.

The derived buffer contains log10(T) after the original SDK forward unmix.
It is never exponentiated and logged again. Its full-source geometry and
pixel support stay unchanged; only the density coordinate changes. Ordinary
source arrays cannot opt into this domain through a recipe or identity label.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np

from .film_profiles import checked_film, film_config, film_signature
from .immutable_source_v44 import ImmutableSourceCertificate, certificate_for, source_digest


DOMAIN = "native-c41-sdk-unmixed-log-density-v45"
SCHEMA = "pure-rgb-native-film-spatial-v45"
LEAF_STATISTIC = "float64 median of original SDK unmixed log10(T); no exp/log round trip, including even-sized leaves"
LOG_POLICY = "original SDK log_density_open(T), then forward unmix_log_image; no upper clamp or exp/log round trip"
_PRE_PRINT_EV_ADAPTER = "9d54ec77b685fb059c9f7baa75f849a352c684cb2fbfb4b1905219db866276b0"
_FIRST_PRINT_EV_ADAPTER = "359d0a6883950d39dad084308909ffc70723ba1b96e37fd36b095168926ecda1"
_DOMAINS = {}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _backing(source):
    while isinstance(source, np.ndarray):
        source = source.base
    return source


def film_domain_for(source):
    if not isinstance(source, np.ndarray):
        return None
    backing = _backing(source)
    reference = _DOMAINS.get(id(backing))
    domain = reference() if reference else None
    if domain is None:
        return None
    expected = domain.source
    if (domain.closed or backing is not _backing(expected)
            or source.dtype != expected.dtype or source.shape != expected.shape
            or source.strides != expected.strides or source.flags.writeable
            or source.__array_interface__["data"][0] != expected.__array_interface__["data"][0]):
        raise ValueError("Film log-domain source binding changed")
    return domain


class FilmDensityDomain:
    def __init__(self, source, config, film):
        import negpy_core
        from negpy.features.exposure.normalization import effective_crosstalk_matrix, unmix_log_image
        from negpy.features.process.models import ProcessMode

        if (not isinstance(source, np.ndarray) or source.dtype != np.float32
                or source.ndim != 3 or source.shape[-1] != 3
                or not source.flags.c_contiguous or source.flags.writeable):
            raise ValueError("Film density domain requires unchanged read-only original F32 RGB")
        if film_domain_for(source) is not None or config.process.process_mode != ProcessMode.C41 or config.process.positive_source:
            raise ValueError("C41 film calibration requires original negative sensor bands, not another derived domain")
        self._original_certificate = certificate_for(source)
        if self._original_certificate is None and not np.isfinite(source).all():
            raise ValueError("Film density domain requires finite original sensor bands")
        self.original = source
        self._original_binding = source
        self._original_layout = (source.shape, source.strides, source.dtype.str,
                                 source.__array_interface__["data"][0])
        self.original_sha256 = source_digest(source)
        self.film = checked_film(film)
        calibrated = film_config(config, self.film)
        self.signature = film_signature(calibrated.process)
        if config.process.crosstalk_strength != 0 and film_signature(config.process) != self.signature:
            raise ValueError("Film calibration differs from the bound Native configuration")
        matrix = effective_crosstalk_matrix(calibrated.process, calibrated.process.process_mode)
        resolved = np.eye(3, dtype=np.float32) if matrix is None else np.asarray(matrix, dtype=np.float32)
        self.matrix = np.frombuffer(resolved.tobytes(), dtype=np.float32).reshape(3, 3)
        self._film_text = _canonical(self.film)
        self._signature_text = _canonical(self.signature)
        self._matrix_bytes = self.matrix.tobytes()
        log = unmix_log_image(negpy_core.log_density_open(source), matrix)
        self.certificate = ImmutableSourceCertificate.create(np.ascontiguousarray(log, dtype=np.float32))
        if self._original_certificate is not None:
            self.certificate.checkpoint = self._original_certificate.checkpoint
        self.source = self.certificate.source
        self.closed = False
        self.neutral_config = replace(config, process=replace(
            config.process, crosstalk_strength=0.0, crosstalk_matrix=None,
            crosstalk_profile="Generic C41", crosstalk_process="Color Negative"))
        self._neutral_config_text = _canonical(self.neutral_config.to_dict())
        self._registry_key = id(_backing(self.source))
        _DOMAINS[self._registry_key] = weakref.ref(self)
        try:
            self.validate()
        except BaseException:
            self.close()
            raise

    def validate(self):
        if self._original_certificate is not None:
            self._original_certificate.validate(self.original)
        elif not np.isfinite(self.original).all():
            raise ValueError("Film original sensor bands became nonfinite")
        layout = (self.original.shape, self.original.strides, self.original.dtype.str,
                  self.original.__array_interface__["data"][0])
        if (self.closed or self.original is not self._original_binding
                or self.original.flags.writeable or layout != self._original_layout
                or source_digest(self.original) != self.original_sha256):
            raise ValueError("Original source changed during film density processing")
        if (_canonical(self.film) != self._film_text or _canonical(self.signature) != self._signature_text
                or self.matrix.tobytes() != self._matrix_bytes or self.matrix.flags.writeable
                or _canonical(self.neutral_config.to_dict()) != self._neutral_config_text):
            raise ValueError("Film density matrix or provenance changed")
        self.certificate.validate(self.source)

    def close(self):
        if not self.closed:
            _DOMAINS.pop(self._registry_key, None)
            self.certificate.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _runtime():
    import negpy_core
    from negpy.features.exposure import normalization

    return dict(adapter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                normalization=hashlib.sha256(Path(normalization.__file__).read_bytes()).hexdigest(),
                kernel=hashlib.sha256(Path(negpy_core.__file__).read_bytes()).hexdigest(),
                numpy=np.__version__)


def _packet(domain, config, source_key, inner_snapshot):
    from .frozen_native_cpu import _config_document

    return dict(schema=SCHEMA, version=1, domain=DOMAIN, source_key=source_key,
                source_pixels_sha256=domain.original_sha256, source_shape=list(domain.original.shape),
                input_config=_config_document(config), film=deepcopy(domain.film),
                film_signature=deepcopy(domain.signature), effective_matrix=domain.matrix.tolist(),
                derived_log_sha256=domain.certificate.sha256,
                inner_source_key=_digest([DOMAIN, source_key, domain.original_sha256,
                                          domain.film, domain.certificate.sha256]),
                inner_snapshot=inner_snapshot, runtime=_runtime())


def _load(snapshot, domain, config, source_key):
    from .spatial_workflow_v31 import MAX_BYTES

    if not isinstance(snapshot, str) or len(snapshot.encode()) > MAX_BYTES:
        raise ValueError("Invalid film spatial snapshot")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate film spatial snapshot key")
            result[key] = value
        return result

    data = json.loads(snapshot, object_pairs_hook=unique)
    if not isinstance(data, dict):
        raise ValueError("Film spatial snapshot requires a versioned object")
    expected = _packet(domain, config, source_key, data.get("inner_snapshot"))
    body = {key: value for key, value in data.items() if key != "sha256"}
    candidates = [expected]
    for adapter in (_PRE_PRINT_EV_ADAPTER, _FIRST_PRINT_EV_ADAPTER):
        # These adapters have the same immutable density-domain contract.
        candidates.append({**expected, "runtime": {**expected["runtime"], "adapter": adapter}})
    if (set(data) != set(expected) | {"sha256"} or data.get("sha256") != _digest(body)
            or _canonical(body) not in map(_canonical, candidates)):
        raise ValueError("Film spatial source, matrix, configuration or snapshot changed")
    return data


def encode_film_spatial(processor, source, config, records, roi, document, *, film,
                        source_key, snapshot=None, highlight=None, point=None, print_ev=0.0):
    from .spatial_workflow_v31 import encode_spatial_workflow
    from .frozen_native_cpu import _config_document

    readonly = source if not source.flags.writeable else source.view()
    readonly.setflags(write=False)
    with FilmDensityDomain(readonly, config, film) as domain:
        saved = _load(snapshot, domain, config, source_key) if snapshot is not None else None
        packet = _packet(domain, config, source_key, None)
        payload, export, info, inner = encode_spatial_workflow(
            processor, domain.source, domain.neutral_config, records, roi, document,
            source_key=packet["inner_source_key"], snapshot=saved["inner_snapshot"] if saved else None,
            highlight=highlight, point=point, print_ev=print_ev,
        )
        domain.validate()
        packet["inner_snapshot"] = inner
        packet["sha256"] = _digest(packet)
        info = {**info, "source_pixels_sha256": domain.original_sha256,
                "derived_input_config_sha256": info["input_config_sha256"],
                "input_config_sha256": _digest(_config_document(config)),
                "density_domain": DOMAIN, "film": deepcopy(domain.film),
                "film_signature": deepcopy(domain.signature),
                "effective_matrix": domain.matrix.tolist(),
                "derived_log_sha256": domain.certificate.sha256,
                "original_source_values_changed": False,
                "log_transform_policy": LOG_POLICY,
                "automatic_detection_domain": "original-linear-selected-sensor-bands"}
        text = _canonical(packet)
        from .spatial_workflow_v31 import MAX_BYTES

        if len(text.encode()) > MAX_BYTES:
            raise ValueError("Film spatial snapshot exceeds the supported size")
        return payload, export, info, text


def validate_film_spatial(snapshot, source, config, roi, document, source_key, highlight, film):
    from .spatial_workflow_v31 import validate_spatial_replay

    readonly = source if not source.flags.writeable else source.view()
    readonly.setflags(write=False)
    with FilmDensityDomain(readonly, config, film) as domain:
        data = _load(snapshot, domain, config, source_key)
        validate_spatial_replay(data["inner_snapshot"], domain.source, domain.neutral_config,
                                roi, document, data["inner_source_key"], highlight)
        domain.validate()
