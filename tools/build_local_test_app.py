"""Build a Finder app, optionally with a relocatable runtime."""

import argparse
import json
import os
import platform
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
import runpy
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
FLAT = Path("assets/branding/deliveries/2026-11-17-flat")
SPLASH = Path("assets/branding/deliveries/2026-11-13-png-final/Asset-1@2x.png")
MANUAL = Path("docs/OPERATING_MANUAL.zh-CN.md")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "* 2.*", "usability_check.py")
_IGNORE_PACKAGES = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".DS_Store", "*.pth", "_virtualenv*", "__editable__*",
    "PyInstaller", "pyinstaller*", "_pyinstaller*", "pytest*", "_pytest", "coverage*",
    "ruff*", "ty", "ty-*.dist-info",
)


def ignore_packages(directory, names):
    """Omit local origin metadata and Finder-created duplicate copies."""
    ignored = set(_IGNORE_PACKAGES(directory, names))
    for name in names:
        if name.casefold() == "crosstalk":
            ignored.add(name)  # Includes symlinks to a developer's personal catalog.
            continue
        if re.fullmatch(r"direct_url(?: \d+)?\.json", name):
            ignored.add(name)
            continue
        original = re.sub(r" 2(?=\.|$)", "", name)
        if original != name and original in names:
            ignored.add(name)
    return ignored


def expose_bundled_qt_plugins(plugin_root):
    """Qt skips macOS-hidden plugins. Normalize the staged copy before signing."""
    if not (plugin_root / "platforms/libqcocoa.dylib").is_file():
        raise ValueError("应用运行依赖缺少 Qt Cocoa 平台插件")
    for path in (plugin_root, *plugin_root.rglob("*")):
        flags = path.lstat().st_flags
        if flags & stat.UF_HIDDEN:
            os.chflags(path, flags & ~stat.UF_HIDDEN, follow_symlinks=False)


def replace_qt_packages(packages, replacement):
    """Stage a complete isolated PyQt runtime, without mixing library versions."""
    replacement = Path(replacement).resolve(strict=True)
    if not all((replacement / name).is_file() for name in (
        'PyQt6/QtCore.abi3.so',
        'PyQt6/Qt6/plugins/platforms/libqcocoa.dylib',
    )) or not any((replacement / 'PyQt6').glob('sip*.so')):
        raise ValueError('替换运行库必须包含匹配的 PyQt6、sip 与 Cocoa 插件')
    candidates = [path for path in replacement.iterdir() if path.name.casefold().startswith('pyqt6')]
    if not any(path.name.casefold().startswith('pyqt6_qt6-') for path in candidates):
        raise ValueError('替换运行库缺少 Qt6 包版本记录')
    for path in packages.iterdir():
        if path.name.casefold().startswith('pyqt6'):
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    for path in candidates:
        if path.is_dir():
            shutil.copytree(path, packages / path.name, symlinks=True, ignore=ignore_packages)
        else:
            shutil.copyfile(path, packages / path.name)
    receipt = replacement / 'PURE-Qt-runtime.json'
    if receipt.is_file():
        shutil.copyfile(receipt, packages / receipt.name)


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


def exclude_local_film_profiles(resources):
    """Only the pinned NegPy git archive supplies distributable film matrices."""
    bundled = resources / "Source/vendor/negpy/crosstalk"
    for path in resources.rglob("*.[tT][oO][mM][lL]"):
        if path.parent == bundled:
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(resources):
            path.unlink()
            continue
        if not path.is_file():
            continue
        if "crosstalk" in (part.casefold() for part in path.relative_to(resources).parts[:-1]):
            path.unlink()
            continue
        try:
            matrix = tomllib.loads(path.read_text(encoding="utf-8-sig")).get("matrix")
        except (UnicodeError, tomllib.TOMLDecodeError):
            continue
        if isinstance(matrix, list) and len(matrix) == 3 and all(
            isinstance(row, list) and len(row) == 3 for row in matrix
        ):
            path.unlink()


def copy_file(relative, destination):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)


def build(output, *, portable=False, build_number="15", test_build=False, qt_packages=None):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("此测试版构建要求 Apple Silicon macOS")
    if output.suffix != ".app" or output.exists():
        raise ValueError("请指定一个尚不存在的 .app 路径")
    if not re.fullmatch(r"[1-9][0-9]*", str(build_number)):
        raise ValueError("构建编号必须是正整数")
    if qt_packages is not None and not portable:
        raise ValueError('独立 Qt 运行库仅适用于可独立运行的应用包')
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dependency_path = Path(sysconfig.get_path("platlib"))
    if not all((dependency_path / name).is_dir() for name in ("PyQt6", "negpy_core", "AppKit")):
        raise ValueError("请从项目 vendor/negpy/.venv 的 Python 运行构建")
    if portable and qt_packages is None:
        from PyQt6.QtCore import qVersion

        qt_version = tuple(map(int, qVersion().split('.')))
        if (6, 11, 0) <= qt_version < (6, 12, 1):
            raise ValueError('当前 Qt 运行库受 QTBUG-149612 影响；请用 --qt-packages 指定已验证修复的独立运行库')
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
        (source / "pure_rgb/build-info.json").write_text(json.dumps(dict(
            build=str(build_number), test=bool(test_build),
        )))
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
            if qt_packages is not None:
                replace_qt_packages(resources / 'Packages', qt_packages)
            expose_bundled_qt_plugins(resources / "Packages/PyQt6/Qt6/plugins")
            negpy_pin = runpy.run_path(str(ROOT / "bootstrap.py"))["SOURCES"][0][2]
            public_source = f"https://github.com/marcinz606/NegPy/tree/{negpy_pin}/negpy_core"
            sanitize_sboms(resources / "Packages", public_source)
            copy_file(Path("LICENSE"), source)
        exclude_local_film_profiles(resources)
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump({
                "CFBundleIdentifier": "io.github.easonzhangbin.purergb",
                "CFBundleName": "PURE RGB",
                "CFBundleDisplayName": "PURE RGB 测试版" if test_build else "PURE RGB",
                "CFBundleExecutable": executable.name,
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": release,
                "CFBundleVersion": str(build_number),
                "CFBundleIconFile": "AppIcon.icns",
                "NSHighResolutionCapable": True,
                **({"PURE RGB Test Build": True} if test_build else {}),
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
        viewer = runpy.run_path(str(ROOT / "pure_rgb/light_table_launcher.py"))
        viewer["build_executable"](contents / "Helpers" / viewer["NAME"])
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
    parser.add_argument("--build-number", default="15")
    parser.add_argument("--test-build", action="store_true")
    parser.add_argument('--qt-packages', type=Path, help='已隔离验证的完整 PyQt6 运行库目录')
    args = parser.parse_args()
    build(args.output.resolve(), portable=args.portable,
          build_number=args.build_number, test_build=args.test_build,
          qt_packages=args.qt_packages)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
