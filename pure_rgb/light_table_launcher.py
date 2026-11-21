"""Launch the bundled AppKit light table, or build it once for a source checkout."""

import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

MODULE = Path(__file__).resolve().parent
NAME = "PURE RGB Light Table"
SOURCES = [MODULE / "UltraThinMaterial.swift", *sorted((MODULE / "light_table_native").glob("*.swift"))]
HEADER_LOGO = MODULE.parent / "assets/branding/pure-rgb-header.png"


def bundled_executable():
    # Contents/Resources/Source/pure_rgb/light_table_launcher.py
    contents = MODULE.parent.parent.parent
    if contents.name == "Contents" and contents.parent.suffix == ".app":
        return contents / "Helpers" / NAME
    return None


def launch_command(manifest):
    bundled = bundled_executable()
    if bundled is not None:
        return [str(bundled), str(manifest)]
    return [sys.executable, "-m", "pure_rgb.light_table_launcher", str(manifest)]


def build_executable(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "/usr/bin/xcrun", "swiftc", "-O", "-swift-version", "5",
        "-target", platform.machine() + "-apple-macosx14.0",
        *map(str, SOURCES), "-o", str(output),
    ], check=True)
    subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(output)], check=True)


def development_executable():
    digest = hashlib.sha256(platform.machine().encode())
    for source in [Path(__file__), HEADER_LOGO, *SOURCES]:
        digest.update(source.read_bytes())
    cache = Path.home() / "Library/Caches/PURE RGB/Light Table" / digest.hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    executable = cache / NAME
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not executable.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as scratch:
                output = Path(scratch) / NAME
                build_executable(output)
                shutil.copyfile(HEADER_LOGO, cache / HEADER_LOGO.name)
                output.replace(executable)
    return executable


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("请传入观片台清单路径")
    try:
        helper = bundled_executable() or development_executable()
        os.execv(helper, [str(helper), sys.argv[1]])
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"无法启动原生观片台：{error}")
