"""Public PURE RGB release metadata and stable release comparison."""

import re
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


APP_VERSION = "0.1.0"
GITHUB_REPOSITORY = "easonzhangbin/PURERGB"
GITHUB_URL = f"https://github.com/{GITHUB_REPOSITORY}"
RELEASES_URL = f"{GITHUB_URL}/releases"
LATEST_RELEASE_URL = f"{RELEASES_URL}/latest"
OPERATING_MANUAL_PATH = "docs/OPERATING_MANUAL.zh-CN.md"
OPERATING_MANUAL_URL = f"{GITHUB_URL}/blob/v{APP_VERSION}/{OPERATING_MANUAL_PATH}"
DONATION_URL = "https://github.com/easonzhangbin/PURENeg/blob/main/SUPPORT.md"

_VERSION = re.compile(r"v?(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")


def check_newer_release():
    """Resolve GitHub's latest stable release link and compare its tag."""
    request = Request(
        LATEST_RELEASE_URL, method="HEAD", headers={"User-Agent": f"PURE-RGB/{APP_VERSION}"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            target = urlsplit(response.geturl())
    except HTTPError as error:
        if error.code == 404:
            raise ValueError("GitHub 暂无可查询的正式发布。") from error
        raise ValueError("暂时无法检查更新。") from error
    except OSError as error:
        raise ValueError("暂时无法检查更新。") from error
    prefix = f"/{GITHUB_REPOSITORY}/releases/tag/"
    if target.scheme != "https" or target.netloc != "github.com" or not target.path.startswith(prefix):
        raise ValueError("无法识别 GitHub 返回的版本。")
    match = _VERSION.fullmatch(target.path.removeprefix(prefix))
    if match is None:
        raise ValueError("无法识别 GitHub 返回的版本。")
    latest = tuple(map(int, match.groups()))
    current = tuple(map(int, APP_VERSION.split(".")))
    return ".".join(match.groups()) if latest > current else None
