"""Build a Finder app, optionally with a relocatable runtime."""

import argparse
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
import runpy
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
FLAT = Path("assets/branding/deliveries/2026-11-17-flat")
SPLASH = Path("assets/branding/deliveries/2026-11-13-png-final/Asset-1@2x.png")
MANUAL = Path("docs/OPERATING_MANUAL.zh-CN.md")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "* 2.*")
_IGNORE_PACKAGES = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".DS_Store", "*.pth", "_virtualenv*", "__editable__*",
    "PyInstaller", "pyinstaller*", "_pyinstaller*", "pytest*", "_pytest", "coverage*",
    "ruff*", "ty", "ty-*.dist-info",
)


def ignore_packages(directory, names):
    """Omit local origin metadata and Finder-created duplicate copies."""
    ignored = set(_IGNORE_PACKAGES(directory, names))
    for name in names:
        if re.fullmatch(r"direct_url(?: \d+)?\.json", name):
            ignored.add(name)
            continue
        original = re.sub(r" 2(?=\.|$)", "", name)
        if original != name and original in names:
            ignored.add(name)
    return ignored


def sanitize_sboms(packages, public_source):
    """Remove local editable-build file URIs from the NegPy core SBOM."""
    for sbom in packages.glob("negpy_core-*.dist-info/sboms/*.json"):
        contents = sbom.read_text()
        contents = re.sub(r"file://[^\s\"\\#]+", public_source, contents)
        sbom.write_text(contents)


def relocate_sysconfig(runtime, version, python_base):
    """Resolve Python build-prefix metadata against the bundled runtime."""
    token = "/__pure_rgb_bundled_python__"
    for data in (runtime / "lib" / f"python{version}").glob("_sysconfigdata*.py"):
        contents = data.read_text()
        if str(python_base) not in contents:
            continue
        contents = contents.replace(str(python_base), token)
        contents += (
            "\n# Replace the original build prefix with the actual bundled runtime.\n"
            "import sys as _pure_rgb_sys\n"
            "for _pure_rgb_key, _pure_rgb_value in build_time_vars.items():\n"
            "    if isinstance(_pure_rgb_value, str):\n"
            f"        build_time_vars[_pure_rgb_key] = _pure_rgb_value.replace({token!r}, _pure_rgb_sys.base_prefix)\n"
            "del _pure_rgb_sys, _pure_rgb_key, _pure_rgb_value\n"
        )
        data.write_text(contents)


def bundle_vendors(source, scratch):
    pins = runpy.run_path(str(ROOT / "bootstrap.py"))["SOURCES"]
    local_vendor = Path(sys.prefix).parent.parent
    for name, _, revision in pins:
        checkout = local_vendor / name
        paths = {
            "negpy": ["negpy", "icc", "crosstalk", "gear", "LICENSE", "NOTICE.md", "VERSION"],
            "cool-colors": ["LICENSE", "nikon_cml4.py", "nikon_cml4_oracle.py", "invert_c41.py",
                            "build_builder_luts.py"],
            "flexcolor-tool": ["LICENSE", "flexcolor"],
        }[name]
        archive = scratch / (name + ".tar")
        subprocess.run(["git", "-C", str(checkout), "archive", "-o", str(archive), revision, "--", *paths],
                       check=True)
        destination = source / "vendor" / name
        destination.mkdir(parents=True)
        with tarfile.open(archive) as package:
            package.extractall(destination, filter="data")
        (destination / ".pure-rgb-revision").write_text(revision + "\n")


def copy_file(relative, destination):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)


def build(output, *, portable=False):
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
        if portable:
            relocate_sysconfig(runtime, version, python_base)
            bundle_vendors(source, Path(scratch))
            shutil.copytree(dependency_path, resources / "Packages", symlinks=True, ignore=ignore_packages)
            negpy_pin = runpy.run_path(str(ROOT / "bootstrap.py"))["SOURCES"][0][2]
            public_source = f"https://github.com/marcinz606/NegPy/tree/{negpy_pin}/negpy_core"
            sanitize_sboms(resources / "Packages", public_source)
            copy_file(Path("LICENSE"), source)
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump({
                "CFBundleIdentifier": "io.github.easonzhangbin.purergb",
                "CFBundleName": "PURE RGB",
                "CFBundleDisplayName": "PURE RGB",
                "CFBundleExecutable": executable.name,
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": release,
                "CFBundleVersion": "7",
                "CFBundleIconFile": "AppIcon.icns",
                "NSHighResolutionCapable": True,
                **({"PURE RGB Portable": True, "LSMinimumSystemVersion": "14.0"} if portable else {
                    "PURE RGB Test Workspace": str(ROOT),
                    "PURE RGB Python Dependencies": str(dependency_path),
                }),
                "PURE RGB Python Library": library_name,
            }, stream)
        subprocess.run([
            "/usr/bin/xcrun", "clang", "-fobjc-arc", "-O2", "-Wall", "-Wextra",
            *(["-mmacosx-version-min=14.0"] if portable else []),
            "-framework", "Cocoa", "-o", str(executable), str(ROOT / "tools/local_test_app_launcher.m"),
        ], check=True)
        material_library = source / "pure_rgb/libPUREUltraThinMaterial.dylib"
        subprocess.run([
            "/usr/bin/xcrun", "swiftc", "-emit-library", "-O",
            "-target", "arm64-apple-macosx14.0", "-module-name", "PUREUltraThinMaterial",
            "-Xlinker", "-install_name", "-Xlinker", "@rpath/libPUREUltraThinMaterial.dylib",
            str(ROOT / "pure_rgb/UltraThinMaterial.swift"), "-o", str(material_library),
        ], check=True)
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(material_library)], check=True)
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
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    build(args.output.resolve(), portable=args.portable)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
