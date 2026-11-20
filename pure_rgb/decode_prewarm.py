"""Low-priority, cancellable background preparation of decoded RAW cache."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .diagnostics import event

_GIB = 1024**3


def _fingerprint(manifest_path, manifest):
    """Identify the source and runtime for a previously validated prewarm."""
    from .decoded_cache import _FORMAT, _decoder_stamp
    from .raw_import import file_identity

    try:
        paths = [item["path"] for item in manifest["inputs"]]
        calibration = manifest.get("calibration") or {}
        paths.extend(calibration[name] for name in ("path", "gain_path") if calibration.get(name))
        runtime = [sys.executable, __file__, str(Path(__file__).with_name("decoded_cache.py")),
                   str(Path(__file__).with_name("density_preview_v44.py")),
                   str(Path(__file__).with_name("raw_import.py")),
                   str(Path(__file__).with_name("native.py"))]
        document = {
            "manifest": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "source": [(path, file_identity(path)) for path in paths],
            "runtime": [(path, file_identity(path)) for path in runtime],
            "format": _FORMAT,
            "decoder": _decoder_stamp(),
        }
        if len(manifest["inputs"]) != 3 or not document["decoder"]:
            return None
        return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _index_entry_matches(directory, document):
    from .decoded_cache import _entry_paths
    from .raw_import import file_identity

    try:
        key = document["key"]
        if (type(key) is not str or len(key) != 64
                or any(char not in "0123456789abcdef" for char in key)):
            return False
        data, metadata = _entry_paths(directory, key)
        return (not data.is_symlink() and not metadata.is_symlink()
                and data.is_file() and metadata.is_file()
                and list(file_identity(data)) == document["data_identity"]
                and list(file_identity(metadata)) == document["meta_identity"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def prewarm(manifest_path, cancelled):
    """Populate disk cache in a disposable process, stopping for foreground work."""
    from .workflow_exports import _available_memory
    from .decoded_cache import _budget_bytes, _directory, _enabled

    try:
        manifest = json.loads(Path(manifest_path).read_text())
        shape = manifest["source_shape"]
        if (manifest.get("state") != "complete" or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape)
                or manifest["recipe"].get("source_mode") == "bw-single-v1"):
            return
        if not _enabled() or _budget_bytes() < shape[0] * shape[1] * shape[2] * 4 + 4096:
            return
        directory = _directory()
        fingerprint = _fingerprint(manifest_path, manifest)
        index = directory / ("prewarm-" + fingerprint + ".json") if directory and fingerprint else None
        if index and index.is_file() and not index.is_symlink():
            try:
                if _index_entry_matches(directory, json.loads(index.read_text())):
                    event("decoded_prewarm_hit")
                    return
            except (OSError, ValueError, TypeError):
                pass
        # Account for decoding, alignment and certificate copies while reserving
        # memory for the foreground application. Unknown memory stays idle.
        required = max(6 * _GIB, shape[0] * shape[1] * shape[2] * 4 * 15 + 4 * _GIB)
        available = _available_memory()
        if cancelled() or available is None or available < required:
            return
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
        nice = shutil.which("nice")
        command = [sys.executable, "-m", "pure_rgb.decode_prewarm", str(manifest_path)]
        if nice:
            command = [nice, "-n", "10", *command]
        with tempfile.TemporaryDirectory(prefix="pure-rgb-prewarm-") as scratch:
            command.append(scratch)
            result_path = Path(scratch) / "decoded-result.json"
            environment["PURE_RGB_PREWARM_RESULT"] = str(result_path)
            process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent.parent),
                                       env=environment, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                next_memory_check = time.monotonic() + 1
                stopped = False
                while process.poll() is None:
                    if cancelled():
                        process.terminate()
                        stopped = True
                        break
                    time.sleep(0.25)
                    if time.monotonic() >= next_memory_check:
                        available = _available_memory()
                        if available is None or available < 3 * _GIB:
                            process.terminate()
                            stopped = True
                            break
                        next_memory_check = time.monotonic() + 1
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            event("decoded_prewarm_finished", code=process.returncode)
            if (not stopped and process.returncode == 0 and index and result_path.is_file()
                    and fingerprint == _fingerprint(manifest_path, manifest)):
                result = json.loads(result_path.read_text())
                if _index_entry_matches(directory, result):
                    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".prewarm-marker-",
                                                     delete=False) as stream:
                        temporary = Path(stream.name)
                        try:
                            json.dump(result, stream)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                    try:
                        temporary.replace(index)
                    finally:
                        temporary.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        event("decoded_prewarm_failed", error=str(error))


def main():
    from .density_preview_v44 import DensityPreviewCache
    from .draft_preview import DraftPreview
    from .native import sha256

    # Let a cancelled cache write unwind and remove its temporary files.
    # The parent still enforces a timeout and can kill a stuck native call.
    def terminate(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)

    manifest_path, scratch = sys.argv[1:]
    preview = DraftPreview(manifest_path, sha256(manifest_path), scratch)
    cache = DensityPreviewCache(preview)
    try:
        cache.prepare_source()
    finally:
        cache.close()


if __name__ == "__main__":
    main()
