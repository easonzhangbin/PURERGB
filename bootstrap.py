"""Fetch fixed source revisions and build the existing NegPy environment."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = (
    ("negpy", "marcinz606/NegPy", "d8123a9b2a18fdaab5941347a91e402fa936d3dc"),
    (
        "cool-colors",
        "rohanpandula/cool-colors",
        "74ffceb79e6e976858d9bffddfeefddba626378f",
    ),
    (
        "flexcolor-tool",
        "rohanpandula/flexcolor-tool",
        "04e745a2fde3559acb2d4f194b7a192150b26f5c",
    ),
)
MACOS_FRAMEWORKS = (
    "pyobjc-core==13.0",
    "pyobjc-framework-Cocoa==13.0",
    "pyobjc-framework-Quartz==13.0",
)


def main():
    (ROOT / "vendor").mkdir(exist_ok=True)
    for name, repo, sha in SOURCES:
        dest = ROOT / "vendor" / name
        if not dest.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f"https://github.com/{repo}.git",
                    str(dest),
                ],
                check=True,
            )
            if name == "cool-colors":
                # The public code is MIT. Nikon's captured tables and profile
                # have separate rights and aren't part of the default install.
                subprocess.run(
                    ["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
                    input="/*.py\n/LICENSE\n/README.md\n",
                    cwd=dest,
                    text=True,
                    check=True,
                )
            subprocess.run(["git", "checkout", "--detach", sha], cwd=dest, check=True)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dest, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=dest,
            text=True,
        )
        if actual != sha or dirty:
            raise SystemExit(f"Source mismatch; existing checkout is preserved: {dest}")
    subprocess.run(
        ["uv", "sync", "--frozen", "--group", "dev", "--python", "3.13"],
        cwd=ROOT / "vendor/negpy",
        check=True,
    )
    if sys.platform == "darwin":
        python = ROOT / "vendor/negpy/.venv/bin/python"
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), *MACOS_FRAMEWORKS],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
