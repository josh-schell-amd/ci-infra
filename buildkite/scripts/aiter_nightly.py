#!/usr/bin/env python3
"""Select, install and check the AITER nightly wheel inside a vLLM ROCm image.

Standard library only: nothing else is guaranteed to be in the image.
"""

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote, urldefrag, urljoin

INDEX_URL = "https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/amd-aiter/"
AITER_COMMIT_URL = "https://github.com/ROCm/aiter/commit/"
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")
OUTPUT_DIR = Path("/opt/aiter-nightly")
# One skipped night passes; older means the producer has stopped.
MAX_WHEEL_AGE_DAYS = 2
# vLLM builds these from source; pip must fail rather than replace them.
PINNED_PACKAGES = ("torch", "torchvision", "torchaudio", "triton")

# amd_aiter-0.1.25+rocm7.2.3.7bfe279.d20261004-cp312-cp312-linux_x86_64.whl:
# AITER main at 7bfe279, built 2026-10-04. The version is a next-release guess.
WHEEL_NAME = re.compile(
    r"^amd_aiter-(?P<base_version>[^+-]+)"
    r"\+rocm(?P<rocm>\d+(?:\.\d+)*)(?:a\d+)?"
    r"\.(?P<commit>[0-9a-f]+)\.d(?P<date>\d{8})"
    r"-(?P<python>cp\d+)-(?P=python)-linux_x86_64\.whl$"
)


@dataclass(frozen=True)
class Wheel:
    filename: str
    url: str
    version: str
    base_version: str
    rocm: str
    commit: str
    build_date: datetime.date
    python: str


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(
                value for name, value in attrs if name == "href" and value
            )


def wheel_from_url(url: str) -> Optional[Wheel]:
    """The AITER nightly wheel an absolute URL points at, if it is one."""
    filename = unquote(urldefrag(url).url.rsplit("/", 1)[-1])
    match = WHEEL_NAME.match(filename)
    if not match:
        return None
    return Wheel(
        filename=filename,
        # As linked: `+` stays `%2B` (a literal `+` gets a 403), #sha256 stays.
        url=url,
        version=filename.split("-")[1],
        base_version=match["base_version"],
        rocm=match["rocm"],
        commit=match["commit"],
        build_date=datetime.datetime.strptime(match["date"], "%Y%m%d").date(),
        python=match["python"],
    )


def parse_index(html: str, index_url: str) -> List[Wheel]:
    """Every AITER nightly wheel linked from a PEP 503 index page."""
    parser = _LinkParser()
    parser.feed(html)
    wheels = (wheel_from_url(urljoin(index_url, href)) for href in parser.hrefs)
    return [wheel for wheel in wheels if wheel]


def rocm_release(version_file_text: str) -> str:
    """`7.2.3-49` (release-build) -> `7.2.3`, as wheel names spell it."""
    match = re.match(r"\d+(?:\.\d+)*", version_file_text.strip())
    if not match:
        raise ValueError(f"Unrecognised ROCm version: {version_file_text!r}")
    return match.group(0)


def python_tag(version_info: Tuple[int, int]) -> str:
    return f"cp{version_info[0]}{version_info[1]}"


def _base_version_key(wheel: Wheel) -> Tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", wheel.base_version))


def select_wheel(wheels: List[Wheel], rocm: str, python: str) -> Wheel:
    """Newest build for this ROCm and Python, by date: version labels lag."""
    matching = [w for w in wheels if w.rocm == rocm and w.python == python]
    if not matching:
        available = sorted({f"rocm{w.rocm}/{w.python}" for w in wheels})
        raise LookupError(
            f"No AITER nightly for rocm{rocm}/{python}. "
            f"Available: {', '.join(available) or 'none'}"
        )
    return max(matching, key=lambda w: (w.build_date, _base_version_key(w)))


def fetch(url: str, attempts: int = 3) -> str:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode()
        except OSError as error:
            if attempt == attempts:
                raise
            print(f"Fetching {url} failed ({error}); retrying", file=sys.stderr)
            time.sleep(10 * attempt)


def pinned_constraints(pip_freeze: str) -> List[str]:
    """`name==version` lines holding PINNED_PACKAGES at what the image has."""
    pinned = []
    for line in pip_freeze.splitlines():
        name = line.split("==", 1)[0]
        if "==" in line and name.lower().replace("_", "-") in PINNED_PACKAGES:
            pinned.append(line.strip())
    return pinned


def annotation(wheel: Wheel, installed: dict, modules: int) -> str:
    """The Buildkite annotation for a build whose wheel passed every check."""
    versions = ", ".join(f"{name} `{version}`" for name, version in installed.items())
    return (
        f":crescent_moon: **AITER nightly**: main at "
        f"[`{wheel.commit}`]({AITER_COMMIT_URL}{wheel.commit}), "
        f"built {wheel.build_date.isoformat()} (`{wheel.version}`)\n\n"
        f"Installed over vLLM's image: {versions}. "
        f"All {modules} prebuilt modules load."
    )


def _installed_versions() -> dict:
    # flydsl: AITER pins it and the wheel installs it.
    names = ("amd-aiter", "flydsl") + PINNED_PACKAGES
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


class NoWheel(Exception):
    pass


class StaleWheel(Exception):
    pass


class InstallFailed(Exception):
    pass


class ModulesDoNotLoad(Exception):
    pass


def _image_wheel(wheel_url: str) -> Wheel:
    """The wheel at wheel_url, if it fits this image's ROCm and Python."""
    wheel = wheel_from_url(wheel_url)
    if not wheel:
        raise NoWheel(f"Not an AITER nightly wheel: {wheel_url}")
    return _for_image([wheel])


def _for_image(wheels: List[Wheel]) -> Wheel:
    rocm = rocm_release(ROCM_VERSION_FILE.read_text())
    try:
        return select_wheel(wheels, rocm, python_tag(sys.version_info[:2]))
    except LookupError as error:
        raise NoWheel(str(error)) from error


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def select(index_url: str, today: datetime.date, wheel_url: str = "") -> Wheel:
    """The wheel to install. A pinned wheel is not checked for age."""
    if wheel_url:
        return _image_wheel(wheel_url)
    wheel = _for_image(parse_index(fetch(index_url), index_url))
    age_days = (today - wheel.build_date).days
    if age_days > MAX_WHEEL_AGE_DAYS:
        raise StaleWheel(
            f"Newest AITER nightly for rocm{wheel.rocm}/{wheel.python} is "
            f"{age_days} days old ({wheel.filename}); the producer has stopped "
            f"publishing for this ROCm."
        )
    return wheel


def install(wheel_url: str, output_dir: Path) -> None:
    """Install the wheel with PINNED_PACKAGES held; record it in wheel.json."""
    wheel = _image_wheel(wheel_url)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints = output_dir / "constraints.txt"
    constraints.write_text("\n".join(pinned_constraints(freeze)) + "\n")
    print(f"Holding vLLM's builds:\n{constraints.read_text()}", flush=True)

    # --no-cache-dir: a cached copy would add 500 MB to the layer.
    pip = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--constraint",
            str(constraints),
            wheel.url,
        ],
    )
    if pip.returncode:
        raise InstallFailed(f"pip could not install {wheel.filename}; see the log")

    installed = _installed_versions()
    if installed.get("amd-aiter") != wheel.version:
        raise InstallFailed(
            f"amd-aiter is {installed.get('amd-aiter')} after install, "
            f"expected {wheel.version}"
        )
    print(json.dumps(installed, indent=2))

    record = asdict(wheel) | {
        "build_date": wheel.build_date.isoformat(),
        "installed": installed,
    }
    (output_dir / "wheel.json").write_text(json.dumps(record, indent=2) + "\n")


def load_modules(module_dir: Path) -> int:
    """Load every prebuilt module with all symbols resolved, so one built
    against another torch fails here rather than in a GPU job."""
    import ctypes

    import torch  # noqa: F401  vLLM's libtorch must be the one they bind to

    modules = sorted(module_dir.glob("*.so"))
    if not modules:
        raise ModulesDoNotLoad(f"No prebuilt modules in {module_dir}")
    failures = []
    for module in modules:
        # Before loading, so a crash names the module.
        print(f"loading {module.name}", file=sys.stderr, flush=True)
        try:
            ctypes.CDLL(str(module), mode=os.RTLD_NOW)
        except OSError as error:
            failures.append(f"{module.name}: {error}")
    if failures:
        raise ModulesDoNotLoad(
            f"{len(failures)} of {len(modules)} prebuilt modules do not load "
            "in vLLM's image:\n" + "\n".join(failures)
        )
    return len(modules)


def check(output_dir: Path) -> str:
    """Load the installed modules; the annotation for a build that passes."""
    spec = importlib.util.find_spec("aiter")  # locates, does not import, aiter
    modules = load_modules(Path(spec.submodule_search_locations[0]) / "jit")
    record = json.loads((output_dir / "wheel.json").read_text())
    wheel = _image_wheel(record["url"])
    return annotation(wheel, record["installed"], modules)


# 1 stays for unexpected errors, which the pipeline retries once.
EXIT_NO_WHEEL = 10
EXIT_STALE_WHEEL = 11
EXIT_INSTALL_FAILED = 12
EXIT_MODULES_DO_NOT_LOAD = 13
EXIT_CODES = {
    NoWheel: EXIT_NO_WHEEL,
    StaleWheel: EXIT_STALE_WHEEL,
    InstallFailed: EXIT_INSTALL_FAILED,
    ModulesDoNotLoad: EXIT_MODULES_DO_NOT_LOAD,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser(
        "select", help="print the URL of the wheel to install"
    )
    select_parser.add_argument("--index-url", default=INDEX_URL)
    select_parser.add_argument(
        "--wheel-url", default="", help="install this wheel; skips the index"
    )
    install_parser = commands.add_parser("install", help="install a selected wheel")
    install_parser.add_argument("--wheel-url", required=True)
    install_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    check_parser = commands.add_parser(
        "check", help="load the installed modules; print the annotation"
    )
    check_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            print(select(args.index_url, _today(), args.wheel_url).url)
        elif args.command == "install":
            install(args.wheel_url, args.output_dir)
        else:
            print(check(args.output_dir))
    except tuple(EXIT_CODES) as error:
        # stdout, in place of the result: the swap script annotates with it.
        print(error)
        return EXIT_CODES[type(error)]
    return 0


if __name__ == "__main__":
    sys.exit(main())
