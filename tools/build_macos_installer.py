"""Package a verified PURE RGB.app for Apple's graphical Installer."""

import argparse
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
import xml.etree.ElementTree as ET


APP_ID = "io.github.easonzhangbin.purergb"


def write_zip_assistant(app, output):
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", app], check=True)
    signed = subprocess.run(["/usr/bin/codesign", "-dv", "--verbose=4", app],
                            check=True, capture_output=True, text=True).stderr
    match = re.search(r"^CDHash=([0-9a-f]{40})$", signed, re.MULTILINE)
    if not match:
        raise ValueError("签名未提供 CodeDirectory 摘要")
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    if info["CFBundleIdentifier"] != APP_ID:
        raise ValueError("安装助手只能用于 PURE RGB")
    template = Path(__file__).with_name("macos_zip_assistant.command.in").read_text()
    version = info["CFBundleShortVersionString"]
    if info.get("PURE RGB Test Build"):
        version += f" 测试版 Build {info['CFBundleVersion']}"
        template = template.replace('default button "在访达中安装"', 'default button "直接打开"')
        template = template.replace("请从 GitHub 正式发布页重新下载并核对 SHA-256。", "请重新解压本次测试包并核对 SHA-256。")
        template = template.replace("从 GitHub 正式发布页核对下载文件的 SHA-256", "核对本次测试包的 SHA-256")
        template = template.replace("GitHub Releases 的 SHA-256", "本次测试包的 SHA-256")
    script = template.replace("@CDHASH@", match[1]).replace("@VERSION@", version)
    with output.open("x") as file:
        file.write(script)
    output.chmod(0o755)


def build_installer(app, output):
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    if app.name != "PURE RGB.app" or info["CFBundleIdentifier"] != APP_ID or not info.get("PURE RGB Portable"):
        raise ValueError("请选择可以独立运行的 PURE RGB.app 正式构建")
    if output.suffix != ".pkg" or output.exists():
        raise ValueError("请指定尚不存在的 .pkg 文件")
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", app], check=True)
    version = info["CFBundleShortVersionString"]
    minimum_os = info["LSMinimumSystemVersion"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build payloads outside File Provider / iCloud folders. Their Finder
    # metadata must not be archived inside the signed application bundle.
    with TemporaryDirectory(prefix="pure-rgb-installer-") as temp:
        work = Path(temp)
        root = work / "root"
        root.mkdir()
        staged_app = root / app.name
        subprocess.run(["/usr/bin/ditto", "--norsrc", "--noextattr", app, staged_app], check=True)
        subprocess.run(["/usr/bin/xattr", "-cr", staged_app], check=True)
        subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", staged_app], check=True)
        metadata = work / "components.plist"
        metadata.write_bytes(plistlib.dumps([dict(
            RootRelativeBundlePath=app.name, BundleIsRelocatable=False,
            BundleIsVersionChecked=True, BundleHasStrictIdentifier=True, BundleOverwriteAction="upgrade",
        )]))
        payload = work / "application.pkg"
        subprocess.run([
            "/usr/bin/pkgbuild", "--root", root, "--component-plist", metadata,
            "--install-location", "/Applications", "--identifier", APP_ID,
            "--version", version, payload,
        ], check=True)
        requirements = work / "requirements.plist"
        requirements.write_bytes(plistlib.dumps({"os": [minimum_os], "arch": ["arm64"]}))
        distribution = work / "Distribution.xml"
        subprocess.run([
            "/usr/bin/productbuild", "--synthesize", "--product", requirements,
            "--package", payload, distribution,
        ], check=True)
        tree = ET.parse(distribution)
        spec = tree.getroot()
        title = f"PURE RGB {version} 安装助手"
        ET.SubElement(spec, "title").text = title
        resources = work / "resources"
        resources.mkdir()
        (resources / "Welcome.txt").write_text(
            f"{title}\n\n本版本适用于 Apple 芯片 Mac，要求 macOS {minimum_os} 或更高版本。\n\n"
            "请先保存任务并退出 PURE RGB。点击「继续」可查看安装位置并确认安装。"
            "程序将安装到「应用程序」，照片和项目保存在原处。\n", encoding="utf-8",
        )
        (resources / "Conclusion.txt").write_text(
            "安装完成。请到「应用程序」打开 PURE RGB。已有任务可在启动时选择载入。\n\n"
            "如果 macOS 提示无法验证开发者，请核对 GitHub Releases 提供的 SHA-256，"
            "再根据 Apple「在 Mac 上安全地打开 App」说明决定是否允许打开。\n"
            "https://support.apple.com/zh-cn/102445\n", encoding="utf-8",
        )
        ET.SubElement(spec, "welcome", file="Welcome.txt", **{"mime-type": "text/plain"})
        ET.SubElement(spec, "conclusion", file="Conclusion.txt", **{"mime-type": "text/plain"})
        tree.write(distribution, encoding="utf-8", xml_declaration=True)
        product = work / "installer.pkg"
        subprocess.run([
            "/usr/bin/productbuild", "--distribution", distribution, "--resources", resources,
            "--package-path", work, "--version", version, "--identifier", f"{APP_ID}.installer", product,
        ], check=True)
        # Publish on the destination filesystem without overwriting a release.
        with NamedTemporaryFile(prefix=".pure-rgb-installer-", dir=output.parent) as staged:
            with product.open("rb") as stream:
                shutil.copyfileobj(stream, staged)
            os.fchmod(staged.fileno(), 0o644)
            staged.flush()
            os.fsync(staged.fileno())
            os.link(staged.name, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zip-assistant", type=Path,
                        help="输出放在 ZIP 中 PURE RGB.app 旁的双击安装助手")
    args = parser.parse_args()
    build_installer(args.app.resolve(strict=True), args.output.resolve())
    if args.zip_assistant:
        write_zip_assistant(args.app, args.zip_assistant)
