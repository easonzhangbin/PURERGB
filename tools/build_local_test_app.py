"""Build a local Finder app using the current PURE RGB source checkout."""

import argparse
import platform
import plistlib
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
FLAT = Path("assets/branding/deliveries/2026-11-17-flat")
SPLASH = Path("assets/branding/deliveries/2026-11-13-png-final/Asset-1@2x.png")
MANUAL = Path("docs/OPERATING_MANUAL.zh-CN.md")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "* 2.*")


def copy_file(relative, destination):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)


def build(output):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("此测试版构建要求 Apple Silicon macOS")
    if output.suffix != ".app" or output.exists():
        raise ValueError("请指定一个尚不存在的 .app 路径")
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dependency_path = Path(sysconfig.get_path("platlib"))
    if not all((dependency_path / name).is_dir() for name in ("PyQt6", "negpy_core", "AppKit")):
        raise ValueError("请从项目 vendor/negpy/.venv 的 Python 运行构建")
    python_base = Path(sys.base_prefix)
    library_name = f"lib/libpython{version}.dylib"
    release = runpy.run_path(str(ROOT / "pure_rgb/release.py"))["APP_VERSION"]
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="pure-rgb-app-") as scratch:
        bundle = Path(scratch) / output.name
        contents = bundle / "Contents"
        executable = contents / "MacOS/PURE RGB"
        executable.parent.mkdir(parents=True)
        resources = contents / "Resources"
        source = resources / "Source"
        runtime = resources / "Python"
        shutil.copytree(ROOT / "pure_rgb", source / "pure_rgb", ignore=IGNORE, copy_function=shutil.copyfile)
        shutil.copytree(ROOT / FLAT, source / FLAT, ignore=IGNORE, copy_function=shutil.copyfile)
        shutil.copytree(ROOT / "assets/ui", source / "assets/ui", ignore=IGNORE, copy_function=shutil.copyfile)
        copy_file(Path("assets/branding/pure-rgb-header.png"), source)
        copy_file(Path("assets/branding/originals/pureneg-wordmark.png"), source)
        copy_file(SPLASH, source)
        copy_file(MANUAL, source)
        shutil.copytree(ROOT / "docs/images", source / "docs/images",
                        ignore=IGNORE, copy_function=shutil.copyfile)
        copy_file(Path("THIRD_PARTY_NOTICES.md"), source)
        shutil.copytree(ROOT / "licenses", source / "licenses", ignore=IGNORE, copy_function=shutil.copyfile)
        shutil.copyfile(ROOT / FLAT / "app.icns", resources / "AppIcon.icns")
        shutil.copytree(python_base / "lib" / f"python{version}", runtime / "lib" / f"python{version}",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "site-packages"),
                        copy_function=shutil.copyfile)
        shutil.copyfile(python_base / library_name, runtime / library_name)
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump({
                "CFBundleIdentifier": "io.github.easonzhangbin.purergb",
                "CFBundleName": "PURE RGB",
                "CFBundleDisplayName": "PURE RGB",
                "CFBundleExecutable": executable.name,
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": release,
                "CFBundleVersion": "1",
                "CFBundleIconFile": "AppIcon.icns",
                "NSHighResolutionCapable": True,
                "PURE RGB Test Workspace": str(ROOT),
                "PURE RGB Python Dependencies": str(dependency_path),
                "PURE RGB Python Library": library_name,
            }, stream)
        subprocess.run([
            "/usr/bin/xcrun", "clang", "-fobjc-arc", "-O2", "-Wall", "-Wextra",
            "-framework", "Cocoa", "-o", str(executable), str(ROOT / "tools/local_test_app_launcher.m"),
        ], check=True)
        # Finder metadata from locally copied directories prevents signing.
        subprocess.run(["/usr/bin/xattr", "-cr", str(bundle)], check=True)
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(bundle)], check=True)
        subprocess.run(["/usr/bin/codesign", "--verify", "--strict", str(bundle)], check=True)
        shutil.move(bundle, output)
        subprocess.run(["/usr/bin/xattr", "-cr", str(output)], check=True)
        subprocess.run(["/usr/bin/codesign", "--verify", "--strict", str(output)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
