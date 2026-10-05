import datetime
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from buildkite.scripts import aiter_nightly

INDEX_URL = "https://example.com/whl-multi-arch/amd-aiter/"

# The real index listing on 2026-09-29.
LISTED = """
amd_aiter-0.1.23+rocm7.14.0.136de2b.d20260921-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.1ad6016.d20260923-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.76cd9af.d20260914-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.9252f46.d20260915-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.ba5f08e.d20260916-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.c15adc7.d20260919-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.e2d019f.d20260925-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.14.0.f8785ca.d20260920-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.11d3852.d20260924-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.136de2b.d20260921-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.1ad6016.d20260923-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.49c6fdd.d20260922-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.76cd9af.d20260914-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.9252f46.d20260915-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.ba5f08e.d20260916-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.c15adc7.d20260919-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.e2d019f.d20260925-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.23+rocm7.2.3.f8785ca.d20260920-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.24+rocm10.1.0a20260910.6176e34.d20260926-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.24+rocm10.1.0a20260910.8253efc.d20260927-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.24+rocm7.14.0.6176e34.d20260926-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.24+rocm7.14.0.8253efc.d20260927-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.24+rocm7.2.3.6176e34.d20260926-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.24+rocm7.2.3.8253efc.d20260927-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.25+rocm10.1.0a20260910.bcb56d9.d20260928-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.25+rocm7.14.0.bcb56d9.d20260928-cp312-cp312-linux_x86_64.whl
amd_aiter-0.1.25+rocm7.2.3.bcb56d9.d20260928-cp312-cp312-linux_x86_64.whl
""".split()


def _index(filenames):
    links = "\n".join(f'<a href="{name}">{name}</a><br/>' for name in filenames)
    return f"<html><body><h1>Links for amd-aiter</h1>{links}</body></html>"


@pytest.fixture
def listed_wheels():
    return aiter_nightly.parse_index(_index(LISTED), INDEX_URL)


def test_parses_every_listed_wheel(listed_wheels):
    assert len(listed_wheels) == len(LISTED)
    wheel = next(w for w in listed_wheels if w.filename == LISTED[-1])
    assert wheel.version == "0.1.25+rocm7.2.3.bcb56d9.d20260928"
    assert wheel.rocm == "7.2.3"
    assert wheel.commit == "bcb56d9"
    assert wheel.build_date == datetime.date(2026, 9, 28)
    assert wheel.python == "cp312"
    assert wheel.url == INDEX_URL + LISTED[-1]


def test_commit_starting_with_digits_is_not_read_as_rocm(listed_wheels):
    wheel = next(w for w in listed_wheels if w.commit == "11d3852")
    assert wheel.rocm == "7.2.3"


def test_rocm_prerelease_stamp_is_dropped(listed_wheels):
    assert {w.rocm for w in listed_wheels} == {"7.2.3", "7.14.0", "10.1.0"}


@pytest.mark.parametrize(
    ("rocm", "commit"),
    [("7.2.3", "bcb56d9"), ("7.14.0", "bcb56d9"), ("10.1.0", "bcb56d9")],
)
def test_selects_newest_build_for_rocm(listed_wheels, rocm, commit):
    wheel = aiter_nightly.select_wheel(listed_wheels, rocm, "cp312")
    assert (wheel.rocm, wheel.commit) == (rocm, commit)
    assert wheel.build_date == datetime.date(2026, 9, 28)


def test_build_date_beats_version():
    wheels = aiter_nightly.parse_index(
        _index(
            [
                "amd_aiter-0.1.24+rocm7.2.3.aaaaaaa.d20260926-cp312-cp312-linux_x86_64.whl",
                "amd_aiter-0.1.23+rocm7.2.3.bbbbbbb.d20260927-cp312-cp312-linux_x86_64.whl",
            ]
        ),
        INDEX_URL,
    )
    assert aiter_nightly.select_wheel(wheels, "7.2.3", "cp312").commit == "bbbbbbb"


def test_skipped_night_selects_last_published(listed_wheels):
    # rocm7.14.0 has no 09-22 or 09-24 build.
    older = [w for w in listed_wheels if w.build_date <= datetime.date(2026, 9, 24)]
    wheel = aiter_nightly.select_wheel(older, "7.14.0", "cp312")
    assert wheel.build_date == datetime.date(2026, 9, 23)


@pytest.mark.parametrize(("rocm", "python"), [("7.3.0", "cp312"), ("7.2.3", "cp313")])
def test_no_matching_wheel_fails_with_what_exists(listed_wheels, rocm, python):
    with pytest.raises(LookupError, match=f"rocm{rocm}/{python}") as error:
        aiter_nightly.select_wheel(listed_wheels, rocm, python)
    assert "rocm7.2.3/cp312" in str(error.value)


def test_encoded_and_hashed_links():
    filename = LISTED[-1]
    href = f"../../packages/{filename.replace('+', '%2B')}#sha256=abc123"
    wheels = aiter_nightly.parse_index(
        _index([]).replace("</h1>", f'</h1><a href="{href}">x</a>'), INDEX_URL
    )
    assert [w.filename for w in wheels] == [filename]
    assert wheels[0].url == (
        "https://example.com/packages/"
        + filename.replace("+", "%2B")
        + "#sha256=abc123"
    )


def test_ignores_other_files():
    wheels = aiter_nightly.parse_index(
        _index(
            ["amd_aiter-0.1.25.tar.gz", "other-1.0-cp312-cp312-linux_x86_64.whl", "../"]
        ),
        INDEX_URL,
    )
    assert wheels == []


@pytest.mark.parametrize(
    ("text", "release"),
    [("7.2.3-49\n", "7.2.3"), ("7.14.0", "7.14.0"), ("10.1.0-20260910", "10.1.0")],
)
def test_rocm_release(text, release):
    assert aiter_nightly.rocm_release(text) == release


def test_rocm_release_rejects_garbage():
    with pytest.raises(ValueError):
        aiter_nightly.rocm_release("unknown")


def test_pinned_constraints_keep_only_vllms_builds():
    freeze = "\n".join(
        [
            "amd-aiter==0.1.21.post2",
            "flydsl==0.3.2",
            "torch==2.12.0a0+git6bbd260",
            "torchvision==0.27.1",
            "triton==3.7.1+gitf0b55c0",
            "triton-kernels==1.0.0",
        ]
    )
    assert aiter_nightly.pinned_constraints(freeze) == [
        "torch==2.12.0a0+git6bbd260",
        "torchvision==0.27.1",
        "triton==3.7.1+gitf0b55c0",
    ]


NEWEST = INDEX_URL + LISTED[-1]  # rocm7.2.3, bcb56d9, built 2026-09-28
NEWEST_VERSION = "0.1.25+rocm7.2.3.bcb56d9.d20260928"


@pytest.fixture
def image(monkeypatch, tmp_path):
    """An image on ROCm 7.2.3 and Python 3.12, seeing the 2026-09-29 index."""
    rocm_version = tmp_path / "version"
    rocm_version.write_text("7.2.3-49\n")
    monkeypatch.setattr(aiter_nightly, "ROCM_VERSION_FILE", rocm_version)
    monkeypatch.setattr(aiter_nightly, "python_tag", lambda _: "cp312")
    monkeypatch.setattr(aiter_nightly, "fetch", lambda url: _index(LISTED))
    monkeypatch.setattr(aiter_nightly, "_today", lambda: datetime.date(2026, 9, 29))
    return tmp_path / "out"


def test_selects_newest_wheel_for_the_image(image, capsys):
    assert aiter_nightly.main(["select", "--index-url", INDEX_URL]) == 0
    assert capsys.readouterr().out == NEWEST + "\n"


def test_wheel_from_two_days_ago_is_fresh(image):
    # One skipped night must not fail the run.
    wheel = aiter_nightly.select(INDEX_URL, datetime.date(2026, 9, 30))
    assert wheel.url == NEWEST


def test_older_wheel_is_stale(image, monkeypatch, capsys):
    monkeypatch.setattr(aiter_nightly, "_today", lambda: datetime.date(2026, 10, 1))
    assert aiter_nightly.main(["select", "--index-url", INDEX_URL]) == 11
    out = capsys.readouterr().out
    assert "3 days old" in out
    assert LISTED[-1] in out


def test_no_wheel_for_the_images_rocm(image, capsys):
    aiter_nightly.ROCM_VERSION_FILE.write_text("7.3.0-1\n")
    assert aiter_nightly.main(["select", "--index-url", INDEX_URL]) == 10
    out = capsys.readouterr().out
    assert "No AITER nightly for rocm7.3.0/cp312" in out
    assert "rocm7.2.3/cp312" in out  # names what does exist


def test_retry_keeps_its_wheel_without_reading_the_index(image, monkeypatch, capsys):
    # Not the newest, and retried long after it would count as stale.
    pinned = INDEX_URL + next(n for n in LISTED if "rocm7.2.3.8253efc" in n)
    monkeypatch.setattr(aiter_nightly, "fetch", lambda url: pytest.fail("read index"))
    monkeypatch.setattr(aiter_nightly, "_today", lambda: datetime.date(2026, 12, 1))
    assert aiter_nightly.main(["select", "--wheel-url", pinned]) == 0
    assert capsys.readouterr().out == pinned + "\n"


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (
            INDEX_URL + next(n for n in LISTED if "rocm7.14.0.bcb56d9" in n),
            "No AITER nightly for rocm7.2.3/cp312",
        ),
        (
            INDEX_URL + "other-1.0-cp312-cp312-linux_x86_64.whl",
            "Not an AITER nightly wheel",
        ),
    ],
)
def test_pinned_wheel_must_fit_the_image(image, capsys, url, message):
    assert aiter_nightly.main(["select", "--wheel-url", url]) == 10
    assert message in capsys.readouterr().out


class _FakePip:
    def __init__(self, install_fails=False):
        self.calls = []
        self.install_fails = install_fails

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        failed = self.install_fails and "install" in cmd
        stdout = "torch==2.12.0a0+git6bbd260\namd-aiter==0.1.21.post2\n"
        return subprocess.CompletedProcess(cmd, int(failed), stdout=stdout)


def _install(image):
    return aiter_nightly.main(
        ["install", "--wheel-url", NEWEST, "--output-dir", str(image)]
    )


def test_install_holds_torch_and_records_the_wheel(image, monkeypatch):
    pip = _FakePip()
    monkeypatch.setattr(aiter_nightly.subprocess, "run", pip)
    monkeypatch.setattr(
        aiter_nightly,
        "_installed_versions",
        lambda: {"amd-aiter": NEWEST_VERSION, "torch": "2.12.0a0+git6bbd260"},
    )

    assert _install(image) == 0

    install_cmd = pip.calls[-1]
    assert install_cmd[-1] == NEWEST
    assert "--no-cache-dir" in install_cmd
    constraints = install_cmd[install_cmd.index("--constraint") + 1]
    assert open(constraints).read() == "torch==2.12.0a0+git6bbd260\n"
    record = json.loads((image / "wheel.json").read_text())
    assert record["url"] == NEWEST
    assert record["build_date"] == "2026-09-28"
    assert record["installed"]["torch"] == "2.12.0a0+git6bbd260"


def test_install_fails_when_pip_fails(image, monkeypatch, capsys):
    monkeypatch.setattr(aiter_nightly.subprocess, "run", _FakePip(install_fails=True))
    assert _install(image) == 12
    assert "pip could not install" in capsys.readouterr().out
    assert not (image / "wheel.json").exists()


def test_install_fails_when_the_wheel_did_not_land(image, monkeypatch, capsys):
    monkeypatch.setattr(aiter_nightly.subprocess, "run", _FakePip())
    monkeypatch.setattr(
        aiter_nightly, "_installed_versions", lambda: {"amd-aiter": "0.1.21.post2"}
    )
    assert _install(image) == 12
    assert f"expected {NEWEST_VERSION}" in capsys.readouterr().out
    assert not (image / "wheel.json").exists()


@pytest.fixture
def modules(monkeypatch, tmp_path):
    """An installed aiter/jit; `broken` maps a module to its load error."""
    jit = tmp_path / "site-packages" / "aiter" / "jit"
    jit.mkdir(parents=True)
    for name in ("module_cache.so", "module_attention.so"):
        (jit / name).write_bytes(b"")
    broken = {}
    loaded = []

    def cdll(path, mode):
        loaded.append((Path(path).name, mode))
        if Path(path).name in broken:
            raise OSError(broken[Path(path).name])

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr("ctypes.CDLL", cdll)
    # RTLD_NOW is POSIX-only; give it a value where the tests run elsewhere.
    monkeypatch.setattr(aiter_nightly.os, "RTLD_NOW", 2, raising=False)
    spec = types.SimpleNamespace(submodule_search_locations=[str(jit.parent)])
    monkeypatch.setattr(aiter_nightly.importlib.util, "find_spec", lambda _: spec)
    return types.SimpleNamespace(dir=jit, broken=broken, loaded=loaded)


def test_loads_every_module_resolving_all_symbols(modules):
    assert aiter_nightly.load_modules(modules.dir) == 2
    assert sorted(modules.loaded) == [
        ("module_attention.so", aiter_nightly.os.RTLD_NOW),
        ("module_cache.so", aiter_nightly.os.RTLD_NOW),
    ]


def test_a_module_that_does_not_load_names_itself_and_the_symbol(modules):
    symbol = "undefined symbol: _ZN3c106detail14torchCheckFail"
    modules.broken["module_attention.so"] = symbol
    with pytest.raises(aiter_nightly.ModulesDoNotLoad) as error:
        aiter_nightly.load_modules(modules.dir)
    message = str(error.value)
    assert "1 of 2 prebuilt modules do not load" in message
    assert f"module_attention.so: {symbol}" in message
    assert "module_cache.so" not in message


def test_no_modules_is_a_failure_not_a_pass(modules):
    for module in modules.dir.iterdir():
        module.unlink()
    with pytest.raises(aiter_nightly.ModulesDoNotLoad, match="No prebuilt modules"):
        aiter_nightly.load_modules(modules.dir)


def _record(image):
    image.mkdir()
    (image / "wheel.json").write_text(
        json.dumps({"url": NEWEST, "installed": {"amd-aiter": NEWEST_VERSION}})
    )


def test_check_prints_the_annotation_for_a_passing_build(image, modules, capsys):
    _record(image)
    assert aiter_nightly.main(["check", "--output-dir", str(image)]) == 0
    note = capsys.readouterr().out
    assert "https://github.com/ROCm/aiter/commit/bcb56d9" in note
    assert "built 2026-09-28" in note
    assert f"amd-aiter `{NEWEST_VERSION}`" in note
    assert "All 2 prebuilt modules load" in note


def test_check_fails_when_modules_do_not_load(image, modules, capsys):
    _record(image)
    modules.broken["module_cache.so"] = "undefined symbol: x"
    assert aiter_nightly.main(["check", "--output-dir", str(image)]) == 13
    assert "1 of 2 prebuilt modules do not load" in capsys.readouterr().out
